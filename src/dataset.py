import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike

import pandas as pd


logger = logging.getLogger(__name__)


GMT_PATH = os.path.join(DEFAULT_DATA_DIR, "gmt")

GMT_DATA_FILES = [
    "sp_en_trans.csv",
    "cities.csv",
    "smaller_than.csv",
    "larger_than.csv",
    "common_claim_true_false.csv",
    "companies_true_false.csv",
    "counterfact_true_false.csv",
]


@dataclass(frozen=False)
class GMTDataset(DataClassJsonMixin):
    name: str
    examples: list[tuple[str, bool]] = field(default_factory=list)
    _few_shot: list | None = None
    _few_shot_prefix: str | None = None

    def __init__(
        self,
        examples: list[tuple[str, bool]],
        name: str = "GMT Dataset",
        _few_shot: list | None = None,
    ):
        self.examples = examples
        self.name = name
        self._few_shot = _few_shot

        if _few_shot is None:
            self.select_few_shot(3)

        logger.info(f"initialized {self.name} with {len(self.examples)} examples.")

    def select_few_shot(self, n: int):
        if self._few_shot is not None:
            self.examples += self._few_shot
        label_dict = {True: [], False: []}
        for statement, label in self.examples:
            label_dict[label].append((statement, label))

        self._few_shot = []
        n_per_label = n // len(label_dict)
        n_per_label = max(1, n_per_label)

        for label in label_dict:
            if len(self._few_shot) == n:
                break
            label_examples = random.sample(
                label_dict[label], min(n_per_label, len(label_dict[label]))
            )
            self._few_shot.extend(label_examples)

        self.examples = set(self.examples) - set(self._few_shot)
        self.examples = list(self.examples)

        still_needed = n - len(self._few_shot)
        if still_needed > 0:
            self._few_shot.extend(random.sample(self.examples, still_needed))

        self.examples = set(self.examples) - set(self._few_shot)
        self.examples = list(self.examples)

        few_shot_prefix = ""
        for statement, label in self._few_shot:
            statement, label = self.example_to_qa((statement, label))
            few_shot_prefix += f"{statement}{label}\n"

        self._few_shot_prefix = few_shot_prefix

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        statement, label = self.examples[idx]
        statement, label = self.example_to_qa((statement, label))
        return self._few_shot_prefix + statement, label

    @staticmethod
    def example_to_qa(example: tuple[str, bool]) -> tuple[str, str]:
        statement, label = example
        label = " TRUE" if label else " FALSE"
        return f"{statement} This statement is:", label

    @staticmethod
    def from_csv(
        files=str | list[str],
        name: str = "GMT Dataset",
        shuffle: bool = True,
    ):
        if isinstance(files, str):
            files = [files]

        examples = []
        for file in files:
            file_path = os.path.join(GMT_PATH, file)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                examples.append((row["statement"], row["label"] == 1))

        if shuffle:
            random.shuffle(examples)
        return GMTDataset(examples, name)
