from dataclasses import dataclass
from src.utils.typing import ArrayLike, LatentCacheCollection, PathLike
from typing import Literal
import random
import os
import torch
import logging
import json
from src.utils import env_utils

logger = logging.getLogger(__name__)


@dataclass
class ActivationSample:
    activation: ArrayLike
    context: str
    query: str
    label: Literal["yes", "no"]
    layer_name: str | None = None

    def __post_init__(self):
        if isinstance(self.activation, torch.Tensor) == False:
            self.activation = torch.Tensor(self.activation)

        assert self.label in ["yes", "no"]
        assert "#" in self.query


class ActivationLoader:
    def __init__(
        self,
        latent_cache_files: str,
        shuffle: bool = True,
        batch_size: int = 32,
        name: str = "ActivationLoader",
        logging: bool = False,
    ):
        self.latent_cache_files = []
        for file_path in latent_cache_files:
            if os.path.exists(file_path) == False:
                logger.error(f"{file_path} not found")
                continue
            if os.path.isdir(file_path) == True:
                raise logger.error(f"{file_path} should be a json file")
            self.latent_cache_files.append(file_path)

        if shuffle:
            random.shuffle(self.latent_cache_files)

        self.name = name
        self.logging = logging
        self.current_file_idx = 0
        self.buffer: list[ActivationSample] = []
        self.batch_size = batch_size
        self.stop_iteration = False

        with open(
            os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/yes_no.json"), "r"
        ) as f:
            self.YES_NO_PARAPHRASES = json.load(f)

        with open(
            os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/question.json"), "r"
        ) as f:
            self.QUESTION_PARAPHRASES = json.load(f)

    def get_latent_qa(
        self, correct_label, wrong_label, group
    ) -> tuple[str, Literal["yes", "no"]]:
        label = random.choice(["yes", "no"])
        yes_no = random.choice(self.YES_NO_PARAPHRASES)
        question = random.choice(self.QUESTION_PARAPHRASES[group])

        query = "# "
        question = (
            question.format(correct_label)
            if label == "yes"
            else question.format(wrong_label)
        )
        query += question + f" {yes_no}"

        return query, label

    def load_next_file(self):
        if self.current_file_idx >= len(self.latent_cache_files):
            return False

        with open(self.latent_cache_files[self.current_file_idx], "r") as f:
            lcc = LatentCacheCollection.from_json(f.read())

        add_to_buffer = []
        for latent_cache in lcc.latents:
            for layer_name in latent_cache.latents.keys():
                activation = latent_cache.latents[layer_name]
                query, label = self.get_latent_qa(
                    correct_label=latent_cache.correct_label,
                    wrong_label=latent_cache.incorrect_label,
                    group=latent_cache.group,
                )
                add_to_buffer.append(
                    ActivationSample(
                        activation=activation,
                        context=latent_cache.context,
                        query=query,
                        label=label,
                        layer_name=layer_name,
                    )
                )

        if self.logging:
            logger.info(
                f"|> {self.name} <| {self.current_file_idx+1}/{len(self.latent_cache_files)} adding {len(add_to_buffer)} samples. File: {self.latent_cache_files[self.current_file_idx]}"
            )

        random.shuffle(add_to_buffer)
        self.buffer.extend(add_to_buffer)
        self.current_file_idx += 1
        return True if len(self.buffer) > 0 else False

    def next_batch(self):
        if self.stop_iteration:
            raise StopIteration

        if len(self.buffer) < self.batch_size:
            if self.load_next_file() == False:
                self.stop_iteration = True  # will raise StopIteration next time

        # corner case
        if len(self.buffer) == 0:
            raise StopIteration

        batch = self.buffer[: self.batch_size]
        self.buffer = self.buffer[self.batch_size :]

        return batch


def get_batch_paths(
    root: PathLike = os.path.join(env_utils.DEFAULT_RESULTS_DIR, "cached_latents")
):
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(".json"):
                yield os.path.join(root, file)
