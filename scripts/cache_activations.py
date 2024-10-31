import argparse
import logging
import os
import time

import torch
import transformers
from src.functional import get_batch_concept_activations
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from src.utils.typing import LatentCacheCollection
from typing import Optional


logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")

from src.dataset_manager import DatasetManager


def cache_activations(
    model_key: str,
    dataset_group: str | None,
    interested_layer_indices: list[int] = list(range(5, 20)),
    latent_cache_dir: str = "cached_latents",
    batch_size: int = 32,
    limit_samples: Optional[int] = None,
):
    # Initialize model and tokenizer
    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.float32,
    )

    cache_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        latent_cache_dir,
        model_key.split("/")[-1],
    )
    os.makedirs(cache_dir, exist_ok=True)

    dataset_groups = DatasetManager.list_datasets_by_group(dataset_group)
    for group_name in dataset_groups:
        group_dir = os.path.join(cache_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        counter = 0
        dataloader = DatasetManager.from_dataset_group(group_name, batch_size=batch_size)

        if group_name in ["language_identification", "ag_news"]:
            tokenization_kwargs = {
                "padding": "max_length",
                "max_length": 200,
                "truncation": True,
            }
        else:
            tokenization_kwargs = {
                "padding": "longest",
            }

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            prompts = [context_qa.context for context_qa in batch]
            questions = [context_qa.questions for context_qa in batch]
            answers = [context_qa.answers for context_qa in batch]

            latents = get_batch_concept_activations(
                mt=mt,
                prompts=prompts,
                interested_layer_indices=interested_layer_indices,
                check_prediction=None,
                on_token_occur=None,
                tokenization_kwargs=tokenization_kwargs,
            )

            # ! Right now we are not doing any kind of filtering
            # ! If we are doing any kind of runtime filtering then make sure to keep track of which samples got filtered out
            for latent_cache, question, answer in zip(latents, questions, answers):
                latent_cache.questions = question
                latent_cache.answers = [f" {a.strip()}" for a in answer]

            lcc = LatentCacheCollection(latents=latents)
            lcc.detensorize()

            with open(os.path.join(group_dir, f"batch_{batch_idx}.json"), "w") as f:
                f.write(lcc.to_json())

            counter += len(latents)
            if limit_samples is not None and counter >= limit_samples:
                break

        logger.info(
            f"|>> done caching activations for {group_name=} in {group_dir}"
        )

    logger.info(f"cached {counter} samples in {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for caching activations.",
        default="meta-llama/Llama-3.2-3B",
    )
    parser.add_argument(
        "--dataset_group",
        type=str,
        choices=list(DatasetManager.list_datasets_by_group().keys()),
        default=None,  # if specified wll override dataset
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size to use for caching activations.",
        default=32,
    )
    parser.add_argument(
        "--interested_layers",
        type=str,
        help="The layers to cache activations for.",
        default="7-17",
    )
    parser.add_argument(
        "--latent_cache_dir",
        type=str,
        help="The directory to save the latent caches.",
        default="cached_latents",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        help="The maximum number of samples to cache.",
        default=20000,
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)
    logger.info(args)

    interested_layers_range = args.interested_layers.split("-")
    interested_layers = list(
        range(int(interested_layers_range[0]), int(interested_layers_range[1]) + 1)
    )

    cache_activations(
        model_key=args.model,
        dataset_group=args.dataset_group,
        interested_layer_indices=interested_layers,
        batch_size=args.batch_size,
        latent_cache_dir=args.latent_cache_dir,
        limit_samples=args.limit_samples,
    )
