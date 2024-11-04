import argparse
import logging
import os
import random
import shutil
from typing import Literal

import baukit
import torch
import transformers
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb
from src.activation_manager import ActivationLoader, ActivationSample, get_batch_paths
from src.dataset_manager import DatasetManager
from src.functional import free_gpu_cache, get_module_nnsight, interpret_logits
from src.models import ModelandTokenizer
from src.tokens import find_token_range, prepare_input
from src.train_utils import evaluate, get_train_eval_loaders, prepare_batch_input
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


def get_small_validation_set(validate_act_loader: ActivationLoader, num: int = 200):
    """
    gets a smaller random subset of the full validation set. used for logging purposes only.
    """
    cur_eval_batch = []
    cur_eval_loader = ActivationLoader(
        latent_cache_files=random.choices(validate_act_loader.latent_cache_files, k=25),
        batch_size=validate_act_loader.batch_size,
        name="CurEvalLoader",
    )
    while True:
        try:
            cur_eval_batch.extend(cur_eval_loader.next_batch())
        except StopIteration:
            break
    random.shuffle(cur_eval_batch)
    cur_eval_batch = cur_eval_batch[:num]

    return cur_eval_batch


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def patchscope_finetune(
    model_key: str,
    # layers_of_interest: list[int] = list(range(8, 20)),
    checkpoint_save_dir: str = "patchscope",
    num_final_layers_to_tune: int = 10,
    wandb_logging=True,
    batch_size=32,
    cached_latents_dir="cached_latents",
    eval_dataset: str = None,
):
    # Initialize model and tokenizer
    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.float32,
    )
    latent_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        cached_latents_dir,
        model_key.split("/")[-1],
    )

    ############################## Load Activation Loader ##############################
    train_act_loader, id_val_act_loader, ood_act_loader = get_train_eval_loaders(
        latent_dir=latent_dir, ood_dataset_group=eval_dataset, batch_size=batch_size
    )
    ###################################################################################
    checkpoint_save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        checkpoint_save_dir,
        model_key.split("/")[-1],
        eval_dataset,
    )
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    model = mt._model
    model.train()
    ############################## Hyperparameters ##############################
    learning_rate = 5e-5
    log_steps = 100
    checkpoint_interval = 1000
    num_warmup_steps = 1000
    limit_training_steps = 16000
    ############################################################################
    if wandb_logging:
        wandb.init(
            entity="talkative_probes",
            project="talkative_probes",
            name=f"{model_key.split('/')[-1]}_patchscope_{eval_dataset}",
            config={
                "model_key": model_key.split("/")[-1],
                "learning_rate": learning_rate,
                "wandb_log_interval": log_steps,
                "checkpoint_interval": checkpoint_interval,
                "num_warmup_steps": num_warmup_steps,
                "batch_size": batch_size,
            },
        )

    for layer_name in mt.layer_names[:-num_final_layers_to_tune]:
        module = baukit.get_module(model, layer_name)
        for param in module.parameters():
            param.requires_grad = False

    tunable_params = []
    for layer_name in mt.layer_names[-num_final_layers_to_tune:]:
        module = baukit.get_module(model, layer_name)
        for param in module.parameters():
            param.requires_grad = True
            tunable_params.append(param)
        # tunable_params.extend(list(module.parameters()))

    optimizer = torch.optim.AdamW(tunable_params, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=limit_training_steps,
    )
    loss_func = torch.nn.CrossEntropyLoss()

    for step in tqdm(range(limit_training_steps), desc="Training"):
        optimizer.zero_grad()

        try:
            batch = train_act_loader.next_batch()
        except StopIteration:
            logger.info(f"End of training data at step {step + 1}")
            break

        try:
            batch_tokenized, int_tok_idx = prepare_batch_input(batch, mt)
        except Exception as e:
            logger.error(
                f"bad batch, error while tokenizing: {[b.question for b in batch]}"
            )
            logger.info("Skipping this batch.")
            continue

        activations = [b.activation for b in batch]

        with mt.trace(batch_tokenized):
            # replace the latent on all the residual layers
            module_names = mt.layer_names
            for idx, act, int_tok in zip(range(len(batch)), activations, int_tok_idx):
                for module_name in module_names:
                    module = get_module_nnsight(mt, module_name)
                    module.output[0][idx, int_tok, :] = torch.tensor(
                        act, device=mt.device
                    ).to(mt.dtype)

            # output = mt.output.save()
            last_logits = [
                mt.output.logits[idx, -1, :].save() for idx in range(len(batch))
            ]

        last_logits = torch.stack(last_logits)
        batch_labels = [mt.tokenizer(b.label).input_ids[-1] for b in batch]
        batch_labels = torch.tensor(batch_labels, device=mt.device)

        # Cross-entropy loss
        patchscope_loss = loss_func(last_logits, batch_labels)

        # TODO: include natural text and generation loss
        loss = patchscope_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        free_gpu_cache()

        if (step + 1) % log_steps == 0:
            id_eval_batch = get_small_validation_set(id_val_act_loader, 1000)
            ood_eval_batch = get_small_validation_set(ood_act_loader, 1000)
            id_eval_accuracy = evaluate(mt, id_eval_batch)
            ood_eval_accuracy = evaluate(mt, ood_eval_batch)
            log_data = {
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "id_eval_accuracy": id_eval_accuracy,
                "ood_eval_accuracy": ood_eval_accuracy,
            }
            logger.info(f"Step {step + 1}: {log_data}")
            if wandb_logging:
                wandb.log(log_data)

        if (step + 1) % checkpoint_interval == 0:
            if len(os.listdir(checkpoint_save_dir)) > 0:
                last_checkpoint_path = os.path.join(
                    checkpoint_save_dir, os.listdir(checkpoint_save_dir)[-1]
                )
                remove_dir(last_checkpoint_path)

            new_checkpoint_path = os.path.join(
                checkpoint_save_dir, f"checkpoint-{step + 1}"
            )
            model.save_pretrained(new_checkpoint_path)

    ood_validation_accuracy = evaluate(
        mt, ood_act_loader, batch_size, logging_steps=1000
    )
    print("-" * 100)
    id_validation_accuracy = evaluate(
        mt, id_val_act_loader, batch_size, logging_steps=1000
    )
    logger.info(
        f"Finished training.... Validation Accuracy on full set (ID/OOD): {id_validation_accuracy} / {ood_validation_accuracy}"
    )
    # Save the final model
    if len(os.listdir(checkpoint_save_dir)) > 0:
        last_checkpoint_path = os.path.join(
            checkpoint_save_dir, os.listdir(checkpoint_save_dir)[-1]
        )
        remove_dir(last_checkpoint_path)
    logger.info(f"Saving Final Tuned LM | >> {step+1=} << | path={new_checkpoint_path}")
    new_checkpoint_path = os.path.join(checkpoint_save_dir, f"checkpoint-{step + 1}")
    model.save_pretrained(new_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        help="Model key to load.",
        default="meta-llama/Llama-3.2-3B",
    )

    parser.add_argument(
        "--wandb_logging",
        action="store_true",
        help="Enable Weights and Biases logging.",
        default=True,
    )

    parser.add_argument(
        "--checkpoint_save_dir",
        type=str,
        default="patchscope_test",
        help="Directory to save checkpoints in results.",
    )

    parser.add_argument(
        "--num_final_layers_to_tune",
        type=int,
        default=10,
        help="Number of final layers to tune.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        choices=list(DatasetManager.list_datasets_by_group().keys()),
        help="Evaluation dataset group.",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)
    patchscope_finetune(
        model_key=args.model,
        checkpoint_save_dir=args.checkpoint_save_dir,
        num_final_layers_to_tune=args.num_final_layers_to_tune,
        wandb_logging=args.wandb_logging,
        batch_size=args.batch_size,
        eval_dataset=args.eval,
    )
