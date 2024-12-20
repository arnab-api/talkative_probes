import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from typing import Literal

import torch

from src.utils import env_utils
from src.utils.typing import ArrayLike, LatentCacheCollection, PathLike

logger = logging.getLogger(__name__)


@dataclass
class ActivationSample:
    activation: ArrayLike
    context: str
    question: str
    label: Literal[" Yes", " No"]  # " yes" or " no" is how the labels get tokenized
    layer_name: str | None = None

    def __post_init__(self):
        if isinstance(self.activation, torch.Tensor) == False:
            self.activation = torch.Tensor(self.activation)

        assert self.label in [" Yes", " No"]
        assert "#" in self.question


class ActivationLoader:
    def __init__(
        self,
        latent_cache_files: str,
        shuffle: bool = True,
        batch_size: int = 32,
        name: str = "ActivationLoader",
        logging: bool = False,
        device: torch.device | None = None,
    ):
        self.latent_cache_files = []
        for file_path in latent_cache_files:
            if os.path.exists(file_path) == False:
                logger.error(f"{file_path} not found")
                continue
            if os.path.isdir(file_path) == True:
                raise logger.error(f"{file_path} should be a file")
            self.latent_cache_files.append(file_path)

        if shuffle:
            random.shuffle(self.latent_cache_files)

        self.name = name
        self.logging = logging
        self.current_file_idx = 0
        self.buffer: list[ActivationSample] = []
        self.batch_size = batch_size
        self.stop_iteration = False
        self.device = device

        with open(
            os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/yes_no.json"), "r"
        ) as f:
            self.YES_NO_PARAPHRASES = json.load(f)

        with open(
            os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/question.json"), "r"
        ) as f:
            self.QUESTION_PARAPHRASES = json.load(f)

    # def get_latent_qa(
    #     self, correct_label, wrong_label, group
    # ) -> tuple[str, Literal["yes", "no"]]:
    #     assert isinstance(correct_label, str) and isinstance(wrong_label, str)
    #     assert correct_label != wrong_label

    #     label = random.choice(["yes", "no"])
    #     yes_no = random.choice(self.YES_NO_PARAPHRASES)
    #     question = random.choice(self.QUESTION_PARAPHRASES[group])

    #     query = "# "
    #     question = (
    #         question.format(correct_label)
    #         if label == "yes"
    #         else question.format(wrong_label)
    #     )
    #     query += question + f" {yes_no}"

    #     return query, label

    def load_next_file(self):
        if self.current_file_idx >= len(self.latent_cache_files):
            return False

        try:
            file_path = self.latent_cache_files[self.current_file_idx]
            if file_path.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    lcc = pickle.load(f)
            else:
                with open(file_path, "r") as f:
                    lcc = LatentCacheCollection.from_json(f.read())
            lcc.retensorize(device=self.device)
        except Exception as e:
            logger.error(
                f"Bad file in {self.latent_cache_files[self.current_file_idx]}: {e}"
            )
            logger.info(f"skipping to next file")
            self.current_file_idx += 1
            return self.load_next_file()

        add_to_buffer = []
        for latent_cache in lcc.latents:
            for layer_name in latent_cache.latents.keys():
                activation = latent_cache.latents[layer_name]
                # query, label = self.get_latent_qa(
                #     correct_label=latent_cache.correct_label,
                #     wrong_label=latent_cache.incorrect_label,
                #     group=latent_cache.group,
                # )
                question, label = random.choice(
                    list(zip(latent_cache.questions, latent_cache.answers))
                )
                add_to_buffer.append(
                    ActivationSample(
                        activation=activation,
                        context=latent_cache.context,
                        question=question,
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
            if file.endswith(".pkl") or file.endswith(".json"):
                yield os.path.join(root, file)
