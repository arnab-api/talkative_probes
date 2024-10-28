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
    dataset_names: (
        list[tuple[str, str]] | tuple[str, str]
    ),  #! (group_name, dataset_name) => (relations, factual/country_capital)
    interested_layer_indices: list[int] = list(range(5, 20)),
    latent_cache_dir: str = "cached_latents",
    batch_size: int = 32,
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

    for group_name, dataset_name in dataset_names:
        group_dir = os.path.join(cache_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        data_dir = os.path.join(group_dir, dataset_name)
        os.makedirs(data_dir, exist_ok=True)

        dataloader = DatasetManager.from_named_datasets(
            [(group_name, dataset_name)],
            batch_size=batch_size,
        )

        for batch_idx, batch in enumerate(dataloader):
            prompts = [b.context for b in batch]

            #! check `notebooks/test.ipynb` for example usage
            latents = get_batch_concept_activations(
                mt=mt,
                prompts=prompts,
                interested_layer_indices=interested_layer_indices,
                check_prediction=None,  # will check the next token if passed | None if skipped
                on_token_occur=None,  # always get activations at the last position
            )
            lcc = LatentCacheCollection(latents=latents)
            lcc.detensorize()
            with open(os.path.join(data_dir, f"batch_{batch_idx}.json"), "w") as f:
                f.write(lcc.to_json())

        logger.info(
            f"|>> done caching activations for {group_name=} {dataset_name=} in {data_dir}"
        )

    logger.info(f"Finished caching activations for {model_key}")


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
        "--dataset",
        type=str,
        nargs="+",
        help="The dataset to use for caching activations.",
        default=[
            "relations|factual/country_capital_city",
            "sst2|sst2",
            "geometry_of_truth|cities",
        ],
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
        default="5-20",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)
    logger.info(args)

    interested_layers = list(map(int, args.interested_layers.split("-")))
    dataset_names = [tuple(d.split("|")) for d in args.dataset]

    cache_activations(
        model_key=args.model,
        dataset_names=dataset_names,
        interested_layer_indices=interested_layers,
        batch_size=args.batch_size,
    )
