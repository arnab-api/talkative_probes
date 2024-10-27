import argparse
import logging
import os
import time
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from dataclasses import dataclass, field, fields
from src.functional import get_module_nnsight, get_concept_latents, free_gpu_cache
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
import shutil
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import random, json
from src.utils.typing import LatentCacheCollection
from src.utils.typing import ArrayLike
from typing import Literal
from src.utils.typing import LatentCacheCollection
from src.utils import env_utils
import baukit

logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")

from src.dataset import GMTDataset, GMT_DATA_FILES


def remove_dir(path):
    if os.path.exists(path):
        logger.debug(f"Removing directory: {path}")
        shutil.rmtree(path)


@dataclass(frozen=True)
class LatentSample:
    activation: ArrayLike
    prompt: str
    label: Literal[" yes", " no"]

    def __post_init__(self):
        assert self.label in [" yes", " no"]
        assert "#" in self.prompt


class LatentSampleBuffer:
    idx: int = 0

    def __init__(
        self,
        activations: list[LatentSample],
        batch_size: int = 32,
    ):
        self.activations = activations
        self.batch_size = batch_size

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]

    def __iter__(self):
        for i in range(0, len(self.activations), self.batch_size):
            yield self.activations[i : i + self.batch_size]

    def next_batch(self):
        # return self.activations[self.idx : self.idx + self.batch_size]
        if self.idx >= len(self.activations):
            raise StopIteration
        ret = self.activations[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return ret


with open(
    os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/yes_no.json"), "r"
) as f:
    YES_NO_PARAPHRASES = json.load(f)

with open(
    os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/question.json"), "r"
) as f:
    QUESTION_PARAPHRASES = json.load(f)["GMT"]


def get_latent_qa(yes_ans, no_ans) -> tuple[str, Literal[" yes", " no"]]:
    label = random.choice([" yes", " no"])
    ret = "# "
    yes_no = random.choice(YES_NO_PARAPHRASES)
    question = random.choice(QUESTION_PARAPHRASES)
    question = question.format(yes_ans) if label == " yes" else question.format(no_ans)
    ret += question + f" {yes_no}"
    return ret, label


from src.tokens import prepare_input, find_token_range
from src.functional import interpret_logits, get_module_nnsight


def prepare_batch_input(batch: list[LatentSample], mt: ModelandTokenizer):
    batch_prompts = [b.prompt for b in batch]
    batch_tokenized = prepare_input(
        prompts=batch_prompts, tokenizer=mt, return_offset_mapping=True
    )

    int_tok_idx = []
    for idx in range(len(batch)):
        offset_mapping = batch_tokenized["offset_mapping"][idx]
        act_range = find_token_range(
            string=batch[idx].prompt,
            substring="#",
            occurrence=0,
            tokenizer=mt,
            offset_mapping=offset_mapping,
        )
        int_tok_idx.append(act_range[1] - 1)

    batch_tokenized.pop("offset_mapping")

    return batch_tokenized, int_tok_idx


def patchscope_finetune(
    model_key: str,
    layers_of_interest: list[int] = list(range(8, 20)),
    checkpoint_save_dir: str = "patchscope_tuning",
    num_final_layers_to_tune: int = 10,
    wandb_logging=True,
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

    checkpoint_save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, model_key.split("/")[-1], checkpoint_save_dir
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

    logger.info(os.listdir(latent_dir))

    num_steps = 0
    for cached_file_name in os.listdir(latent_dir):
        logger.info("-" * 100)
        with open(os.path.join(latent_dir, cached_file_name), "r") as f:
            dct = json.load(f)
        lcc = LatentCacheCollection.from_dict(dct)
        lcc.retensorize(device=mt.device)
        logger.info(f"Loaded {len(lcc)} samples from {cached_file_name}")

        latent_arr = []
        layers_of_interest = list(range(8, 20))

        for idx in range(len(lcc.latents)):
            latent_cache = lcc.latents[idx]
            prompt = latent_cache.question
            label = latent_cache.answer
            yes_ans = "true" if str(label).lower().strip() == "true" else "false"
            no_ans = "false" if yes_ans == "true" else "true"
            for layer_idx in layers_of_interest:
                layer_name = mt.layer_name_format.format(layer_idx)
                activation = latent_cache.latents[layer_name]
                prompt, label = get_latent_qa(yes_ans, no_ans)
                latent_arr.append(
                    LatentSample(
                        activation=activation,
                        prompt=prompt,
                        label=label,
                    )
                )

        # # debug
        # latent_arr = random.sample(latent_arr, min(len(latent_arr), 1000))

        random.shuffle(latent_arr)
        train_split = int(len(latent_arr) * 0.9)
        train_buffer = LatentSampleBuffer(latent_arr[:train_split])
        val_buffer = LatentSampleBuffer(latent_arr[train_split:])
        logger.info(f"initialized tuning with {len(latent_arr)} examples")
        logger.info(f"train buffer size: {len(train_buffer)}")

        train_buffer_steps = len(train_buffer) // batch_size
        for _ in tqdm(range(train_buffer_steps), desc="Training"):
            optimizer.zero_grad()
            num_steps += 1

            try:
                batch = train_buffer.next_batch()
            except StopIteration:
                train_buffer.idx = 0
                batch = train_buffer.next_batch()

            batch_tokenized, int_tok_idx = prepare_batch_input(batch, mt)
            # logger.info(batch_tokenized.input_ids.shape)
            activations = [b.activation for b in batch]

            with mt.trace(batch_tokenized):
                module_names = (
                    mt.layer_names
                )  # replace the latent on all the residual layers
                for idx, act, int_tok in zip(
                    range(len(batch)), activations, int_tok_idx
                ):
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
            # break
            optimizer.step()
            scheduler.step()

            free_gpu_cache()

            if num_steps % log_steps == 0:
                log_data = {
                    "loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                logger.info(f"Step {num_steps}: {log_data}")
                if wandb_logging:
                    wandb.log(log_data)

            if num_steps % checkpoint_interval == 0:
                if len(os.listdir(checkpoint_save_dir)) > 0:
                    last_checkpoint_path = os.path.join(
                        checkpoint_save_dir, os.listdir(checkpoint_save_dir)[-1]
                    )
                    remove_dir(last_checkpoint_path)

                new_checkpoint_path = os.path.join(
                    checkpoint_save_dir, f"checkpoint-{num_steps}"
                )
                logger.info(
                    f"|>> {num_steps=} <<| Saving checkpoint to {new_checkpoint_path}"
                )
                model.save_pretrained(new_checkpoint_path)

        logger.info(f"Tuned with {cached_file_name}")

        # Validation
        correct_count = 0
        val_label_counts = {label: 0 for label in ["yes", "no"]}
        print("Validating ...")
        for batch in val_buffer:
            batch_tokenized, int_tok_idx = prepare_batch_input(batch, mt)
            activations = [b.activation for b in batch]

            with torch.no_grad():
                with mt.trace(batch_tokenized):
                    module_names = mt.layer_names
                    for idx, act, int_tok in zip(
                        range(len(batch)), activations, int_tok_idx
                    ):
                        for module_name in module_names:
                            module = get_module_nnsight(mt, module_name)
                            module.output[0][idx, int_tok, :] = torch.tensor(
                                act, device=mt.device
                            )
                    last_logits = [
                        mt.output.logits[idx, -1, :].save() for idx in range(len(batch))
                    ]
                    # output = mt.output.save()
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
            for pred, correct in zip(predicted_labels, correct_labels):
                # print(
                #     f"{pred.token.strip().lower() = } | {correct.strip().lower() = } | >> {pred.token.strip().lower() == correct.strip().lower()}"
                # )
                val_label_counts[correct.strip().lower()] += 1
                if pred.token.strip().lower() == correct.strip().lower():
                    correct_count += 1

        logger.info(
            f"Validation accuracy: {correct_count / len(val_buffer):.4f}, evaluated on {len(val_buffer)} samples {val_label_counts}"
        )
        logger.info("-" * 100)

    logger.info("Finished training.")
    # Save the final model
    if len(os.listdir(checkpoint_save_dir)) > 0:
        last_checkpoint_path = os.path.join(
            checkpoint_save_dir, os.listdir(checkpoint_save_dir)[-1]
        )
        remove_dir(last_checkpoint_path)
    logger.info(
        f"Saving Final Tuned LM | >> {num_steps=} << | path={new_checkpoint_path}"
    )
    new_checkpoint_path = os.path.join(checkpoint_save_dir, f"checkpoint-{num_steps}")
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
        default="patchscope",
        help="Directory to save checkpoints in results.",
    )

    parser.add_argument(
        "--num_final_layers_to_tune",
        type=int,
        default=10,
        help="Number of final layers to tune.",
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
    )
