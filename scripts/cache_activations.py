import argparse
import logging
import os
import time
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from src.functional import get_module_nnsight, get_concept_latents
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
import shutil
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import random, json
from src.utils.typing import LatentCacheCollection


logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")

from src.dataset import GMTDataset, GMT_DATA_FILES


def cache_activations(
    model_key: str,
    data_files: list[str] = GMT_DATA_FILES,
):
    # Initialize model and tokenizer
    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.float32,
    )
    interested_layers = mt.layer_names
    latent_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        "cached_latents",
        model_key.split("/")[-1],
    )
    os.makedirs(latent_dir, exist_ok=True)
    for data_file in data_files:
        dataset = GMTDataset.from_csv(files=data_file, name=data_file.split(".")[0])
        dataset.select_few_shot(0)
        queries = [dataset.examples[i] for i in range(len(dataset))]
        latents = get_concept_latents(
            mt=mt,
            queries=queries,
            interested_layers=interested_layers,
            check_answer=False,
        )

        lcc = LatentCacheCollection(latents=latents)
        lcc.detensorize()
        with open(os.path.join(latent_dir, f"{dataset.name}.json"), "w") as f:
            f.write(lcc.to_json())

        logger.info(f"Cached activations for {dataset.name} in {latent_dir}")

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
        "--data",
        type=str,
        nargs="+",
        default=GMT_DATA_FILES,
        help="List of data files to use for caching activations.",
    )
    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)
    logger.info(args)

    cache_activations(args.model, args.data)
