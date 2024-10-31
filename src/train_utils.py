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
from src.dataset_manager import DatasetManager
from typing import Literal

logger = logging.getLogger(__name__)


class RotatingLoader:
    def __init__(self, act_loader: ActivationLoader, batch_size):
        self.samples = []
        self.batch_size = batch_size
        self.current_idx = 0
        while True:
            try:
                self.samples.extend(act_loader.next_batch())
            except StopIteration:
                break
        random.shuffle(self.samples)
    
    def next_batch(self):
        if self.current_idx + self.batch_size >= len(self.samples):
            self.current_idx = 0
            random.shuffle(self.samples)
        return self.samples[self.current_idx : self.current_idx + self.batch_size]

    def get_n(self, n):
        return self.samples[:n]


def prepare_batch_input(batch: list[ActivationSample], mt: ModelandTokenizer):
    """
    tokenizes the questions in the batch and returns the placeholder index
    """
    batch_prompts = [b.question for b in batch]
    batch_tokenized = prepare_input(
        prompts=batch_prompts,
        tokenizer=mt,
        return_offsets_mapping=True,
        padding_side="left",
    )

    int_tok_idx = []
    for idx in range(len(batch)):
        try:
            offset_mapping = batch_tokenized["offset_mapping"][idx]
            act_range = find_token_range(
                string=batch[idx].question,
                substring="#",
                occurrence=0,
                tokenizer=mt,
                offset_mapping=offset_mapping,
            )
            int_tok_idx.append(act_range[1] - 1)
        except:
            logger.error(
                f"can't find '#' in \"{batch[idx].question}\" ==> bad training data"
            )
            first_attn_token = batch_tokenized["attention_mask"].index(1) + 1
            int_tok_idx.append(first_attn_token)

    batch_tokenized.pop("offset_mapping")

    return batch_tokenized, int_tok_idx


def get_train_eval_loaders(
    latent_dir: str, ood_dataset_group: str | None = None, batch_size: int = 32
) -> tuple[ActivationLoader, ActivationLoader]:
    """
    returns ActivationLoaders for training and validation
    """
    assert ood_dataset_group in DatasetManager.list_datasets_by_group()
    ood_act_batch_paths = list(
        get_batch_paths(os.path.join(latent_dir, ood_dataset_group))
    )
    logger.info(f"{len(ood_act_batch_paths)} OOD batch paths will be loaded")
    random.shuffle(ood_act_batch_paths)
    ood_act_loader = ActivationLoader(
        latent_cache_files=ood_act_batch_paths,
        batch_size=batch_size,
        shuffle=True,
        name="OODValidateLoader",
        device=torch.device("cpu"),
    )

    id_act_batch_paths = []
    for group_dir in os.listdir(os.path.join(latent_dir)):
        if group_dir != ood_dataset_group:
            id_act_batch_paths.extend(
                get_batch_paths(os.path.join(latent_dir, group_dir))
            )
    random.shuffle(id_act_batch_paths)
    train_split = int(len(id_act_batch_paths) * 0.8)
    train_act_batch_paths = id_act_batch_paths[:train_split]
    id_val_act_batch_paths = id_act_batch_paths[train_split:]
    logger.info(f"{len(id_val_act_batch_paths)} ID batch paths will be loaded")

    train_act_loader = ActivationLoader(
        latent_cache_files=train_act_batch_paths,
        batch_size=batch_size,
        shuffle=True,
        name="TrainLoader",
        # logging=True,
    )

    id_val_act_loader = ActivationLoader(
        latent_cache_files=id_val_act_batch_paths,
        batch_size=batch_size,
        shuffle=True,
        name="IDValidateLoader",
        device=torch.device("cpu"),
    )

    return train_act_loader, id_val_act_loader, ood_act_loader


################################## EVALUATION ##################################
@torch.inference_mode()
def evaluate_batch(batch: list[ActivationSample], mt: ModelandTokenizer):

    try:
        batch_tokenized, int_tok_idx = prepare_batch_input(batch, mt)
    except Exception as e:
        logger.error(
            f"bad batch, error while tokenizing: {[b.question for b in batch]}"
        )
        logger.info("Skipping this batch.")
        return 0, 1

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
    return correct_count, len(batch)


@torch.inference_mode()
def evaluate(
    mt: ModelandTokenizer,
    eval_set: list[ActivationSample] | ActivationLoader,
    batch_size=32,
    limit_samples: int | None = None,
    logging_steps: int = 10,
):
    correct_count = 0
    total_count = 0
    step = 0
    if isinstance(eval_set, ActivationLoader):
        while True:
            try:
                batch = eval_set.next_batch()
                correct_batch, len_batch = evaluate_batch(batch, mt)
                correct_count += correct_batch
                total_count += len_batch
            except StopIteration:
                break

            if limit_samples is not None and total_count >= limit_samples:
                break

            step += 1
            if step % logging_steps == 0:
                logger.info(
                    f"Accuracy={correct_count/total_count}({correct_count}/{total_count})"
                )

    else:
        for i in tqdm(range(0, len(eval_set), batch_size), desc="Evaluating"):
            batch = eval_set[i : i + batch_size]
            with torch.no_grad():
                correct_batch, len_batch = evaluate_batch(batch, mt)
                correct_count += correct_batch
                total_count += len_batch

            if limit_samples is not None and total_count >= limit_samples:
                break

            step += 1
            if step % logging_steps == 0:
                logger.info(
                    f"Accuracy={correct_count/total_count}({correct_count}/{total_count})"
                )

    return correct_count / total_count


#############################################################################################
