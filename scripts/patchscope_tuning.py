import argparse
import logging
import os
import torch
import transformers
from src.functional import get_module_nnsight, free_gpu_cache, interpret_logits
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
import shutil
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import random
from src.utils import env_utils
import baukit
from src.activation_manager import ActivationLoader, ActivationSample, get_batch_paths
from src.tokens import prepare_input, find_token_range
from src.functional import free_gpu_cache


logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


def prepare_batch_input(batch: list[ActivationSample], mt: ModelandTokenizer):
    batch_prompts = [b.query for b in batch]
    batch_tokenized = prepare_input(
        prompts=batch_prompts, tokenizer=mt, return_offsets_mapping=True
    )

    int_tok_idx = []
    for idx in range(len(batch)):
        try:
            offset_mapping = batch_tokenized["offset_mapping"][idx]
            act_range = find_token_range(
                string=batch[idx].query,
                substring="#",
                occurrence=0,
                tokenizer=mt,
                offset_mapping=offset_mapping,
            )
            int_tok_idx.append(act_range[1] - 1)
        except:
            logger.error(
                f"can't find '#' in \"{batch[idx].query}\" ==> bad training data"
            )
            first_attn_token = batch_tokenized["attention_mask"].index(1) + 1
            int_tok_idx.append(first_attn_token)

    batch_tokenized.pop("offset_mapping")

    return batch_tokenized, int_tok_idx


@torch.inference_mode()
def evaluate_batch(batch: list[ActivationSample], mt: ModelandTokenizer):
    batch_tokenized, int_tok_idx = prepare_batch_input(batch, mt)
    # logger.debug(f"{batch_tokenized.input_ids.shape=}")
    activations = [b.activation for b in batch]

    with mt.trace(batch_tokenized):
        # patch activation at every layer
        module_names = mt.layer_names
        for idx, act, int_tok in zip(range(len(batch)), activations, int_tok_idx):
            for module_name in module_names:
                module = get_module_nnsight(mt, module_name)
                module.output[0][idx, int_tok, :] = torch.tensor(act, device=mt.device)
        last_logits = [mt.output.logits[idx, -1, :].save() for idx in range(len(batch))]
        # output = mt.output.save() # do not save output, save some memory

    last_logits = torch.stack(last_logits)

    predicted_labels = [
        interpret_logits(
            tokenizer=mt,
            logits=last_logits[idx],
            # logits = output.logits[idx, -1, :],
            k=2,
        )[0]
        for idx in range(len(batch))
    ]

    correct_labels = [b.label for b in batch]
    correct_count = 0

    for pred, correct in zip(predicted_labels, correct_labels):
        # print(f"{str(pred)=} | {correct=}")
        if pred.token.strip().lower() == correct.strip().lower():
            correct_count += 1

    free_gpu_cache()
    return correct_count


@torch.inference_mode()
def evaluate(mt: ModelandTokenizer, eval_set: list[ActivationSample], batch_size=32):
    correct_count = 0
    total_count = 0
    for i in tqdm(range(0, len(eval_set), batch_size), desc="Evaluating"):
        batch = eval_set[i : i + batch_size]
        with torch.no_grad():
            correct_count += evaluate_batch(batch, mt)
            total_count += len(batch)
    return correct_count / total_count


def get_validation_set(validate_act_loader: ActivationLoader, num: int = 200):
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
    layers_of_interest: list[int] = list(range(8, 20)),
    checkpoint_save_dir: str = "patchscope_test",
    num_final_layers_to_tune: int = 10,
    wandb_logging=True,
    batch_size=32,
):
    # Initialize model and tokenizer
    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.float32,
    )
    latent_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        "cached_latents",
        model_key.split("/")[-1],
    )

    ############################## Load Activation Loader ##############################
    activation_batch_paths = list(get_batch_paths(latent_dir))
    random.shuffle(activation_batch_paths)

    train_split = int(len(activation_batch_paths) * 0.8)
    train_act_batch_paths = activation_batch_paths[:train_split]
    val_act_batch_paths = activation_batch_paths[train_split:]

    train_act_loader = ActivationLoader(
        latent_cache_files=train_act_batch_paths,
        batch_size=batch_size,
        shuffle=True,
        name="TrainLoader",
        # logging=True,
    )

    validate_act_loader = ActivationLoader(
        latent_cache_files=val_act_batch_paths,
        batch_size=batch_size,
        shuffle=True,
        name="ValidateLoader",
    )

    ############################## Load Activation Loader ##############################

    checkpoint_save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, checkpoint_save_dir, model_key.split("/")[-1]
    )
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    model = mt._model
    model.train()
    ############################## Hyperparameters ##############################
    learning_rate = 5e-5
    log_steps = 50
    checkpoint_interval = 1000
    num_warmup_steps = 1000
    limit_training_steps = 10000
    batch_size = 32
    ############################################################################
    if wandb_logging:
        wandb.init(
            entity="dl-homeworks",
            project="talkative_probes",
            name=f"{model_key.split('/')[-1]}_patchscope_tune",
            config={
                "model_key": model_key.split("/")[-1],
                "learning_rate": learning_rate,
                "wandb_log_interval": log_steps,
                "checkpoint_interval": checkpoint_interval,
                "num_warmup_steps": num_warmup_steps,
                "batch_size": batch_size,
            },
        )

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

        batch_tokenized, int_tok_idx = prepare_batch_input(batch, mt)

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
            cur_eval_batch = get_validation_set(validate_act_loader, 500)
            eval_accuracy = evaluate(mt, cur_eval_batch)
            log_data = {
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "eval_accuracy": eval_accuracy,
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

    logger.info("Finished training.")
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
    )
