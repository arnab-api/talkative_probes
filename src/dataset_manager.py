import os
import math
import random
import csv
import json
from dataclasses import dataclass
from datasets import load_dataset
import logging

from src.utils.env_utils import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class RawExample:
    feature: str
    label: str  # ? Should we make this Literal["yes", "no"] | Literal["positive", "negative"] instead? -- arnab


class DatasetLoader:
    def __init__(self, group, name):
        self.group = group
        self.name = name

    def load(self) -> list[RawExample]:
        raise NotImplementedError


class GeometryOfTruthDatasetLoader(DatasetLoader):
    GROUP_NAME = "geometry_of_truth"
    DATA_FILES_PATH = os.path.join(DEFAULT_DATA_DIR, "gmt")

    DATASET_NAMES = [
        "sp_en_trans",
        "neg_sp_en_trans",
        "cities",
        "neg_cities",
        "smaller_than",
        "larger_than",
        "common_claim_true_false",
        "companies_true_false",
        "counterfact_true_false",
    ]

    def load(self):
        examples = []
        filename = self.name + ".csv"
        with open(os.path.join(self.DATA_FILES_PATH, filename)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                example = RawExample(feature=row["statement"], label=row["label"])
                examples.append(example)

        return examples

    @staticmethod
    def get_all_loaders():
        loaders = []
        for name in GeometryOfTruthDatasetLoader.DATASET_NAMES:
            loaders.append(
                GeometryOfTruthDatasetLoader(
                    GeometryOfTruthDatasetLoader.GROUP_NAME, name
                )
            )
        return loaders


class SstDatasetLoader(DatasetLoader):
    def load(self):
        dataset = load_dataset("stanfordnlp/sst2")["train"]
        result = []
        for sentence, label in zip(dataset["sentence"], dataset["label"]):
            result.append(RawExample(feature=sentence, label=label))
        return result


RELATION_FILES_ROOT = os.path.join(DEFAULT_DATA_DIR, "relations")
RELATION_NAMES = []
for relation_type in os.listdir(RELATION_FILES_ROOT):
    for file_name in os.listdir(os.path.join(RELATION_FILES_ROOT, relation_type)):
        if file_name.endswith(".json"):
            RELATION_NAMES.append(f"{relation_type}/{file_name[:-5]}")


class RelationDatasetLoader(DatasetLoader):

    GROUP_NAME = "relations"
    DATA_FILES_PATH = RELATION_FILES_ROOT
    DATASET_NAMES = RELATION_NAMES

    def load(self):
        relation_type, relation_name = self.name.split("/")
        file_path = os.path.join(
            self.DATA_FILES_PATH, relation_type, f"{relation_name}.json"
        )
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return examples

        examples = []
        with open(file_path, "r") as f:
            data_dict = json.load(f)
            prompt_templates = data_dict["prompt_templates"]
            objects = [sample["object"] for sample in data_dict["samples"]]
            objects = set(objects)
            for sample in data_dict["samples"]:
                template = random.choice(prompt_templates) + " {}."
                examples.append(
                    RawExample(
                        feature=template.format(sample["subject"], sample["object"]),
                        label="1",
                    )
                )
                false_obj = random.choice(list(objects - {sample["object"]}))
                examples.append(
                    RawExample(
                        feature=template.format(sample["subject"], false_obj),
                        label="0",
                    )
                )
        logger.info(f"Loaded {len(examples)} examples from {self.name}.")
        return examples

    @staticmethod
    def get_all_loaders():
        loaders = []
        for name in RelationDatasetLoader.DATASET_NAMES:
            loaders.append(
                RelationDatasetLoader(RelationDatasetLoader.GROUP_NAME, name)
            )
        return loaders


class DatasetManager:
    supported_datasets: dict[str, DatasetLoader] = {
        dataset.name: dataset
        for dataset in (
            GeometryOfTruthDatasetLoader.get_all_loaders()
            + [SstDatasetLoader(group="sst2", name="sst2")]
            + RelationDatasetLoader.get_all_loaders()
        )
    }

    def __init__(self, examples, batch_size, shuffle):
        self.examples = examples
        self.batch_size = batch_size

        if shuffle:
            random.shuffle(self.examples)

    def split(self, proportions):
        assert sum(proportions) <= 1

        start = 0
        end = None
        result = []
        for proportion in proportions:
            end = start + math.ceil(proportion * len(self.examples))
            result.append(
                DatasetManager(self.examples[start:end], self.batch_size, shuffle=False)
            )
            start = end
        return result

    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.examples), self.batch_size):
            yield self.examples[i : i + self.batch_size]

    @staticmethod
    def from_named_datasets(dataset_names, batch_size=1, shuffle=True):
        examples = []
        for name in dataset_names:
            dataset = DatasetManager.supported_datasets[name]
            examples.extend(dataset.load())
        return DatasetManager(examples, batch_size, shuffle)

    @staticmethod
    def from_dataset_group(group, **kwargs):
        datasets = DatasetManager.list_datasets_by_group(group)
        return DatasetManager.from_named_datasets(datasets[group], **kwargs)

    @staticmethod
    def list_datasets_by_group(group=None):
        result = {}
        for name, dataset in DatasetManager.supported_datasets.items():
            if group is not None and dataset.group != group:
                continue
            if dataset.group not in result:
                result[dataset.group] = []
            result[dataset.group].append(name)
        return result

    @staticmethod
    def list_dataset_groups():
        return list(
            set(
                [
                    dataset.group
                    for dataset in DatasetManager.supported_datasets.values()
                ]
            )
        )
