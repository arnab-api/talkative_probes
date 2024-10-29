import json
import os
import math
import random
import csv
import json
from dataclasses import dataclass
from typing import Literal
from datasets import load_dataset
import logging

import src.utils.env_utils as env_utils

logger = logging.getLogger(__name__)

YES_TOKEN = "Yes"
NO_TOKEN = "No"
NUM_QA_PER_SAMPLE = 10


@dataclass
class ContextQASample:
    context: str
    questions: list[str]
    answers: list[str]
    
    def __post_init__(self):
        for q, a in zip(self.questions, self.answers):
            assert a in (YES_TOKEN, NO_TOKEN)
            assert "#" in q


class DatasetLoader:
    def __init__(self, group, name):
        self.group = group
        self.name = name

        with open(
            os.path.join(env_utils.DEFAULT_DATA_DIR, "paraphrases/question.json")
        ) as f:
            self.question_paraphrases = json.load(f)[group]

    # Must be overridden by dataset class
    def load(self) -> list[ContextQASample]:
        raise NotImplementedError


class MdGenderDatasetLoader(DatasetLoader):
    GROUP_NAME = "md_gender"
    DATASET_NAME = "md_gender"

    def __init__(self):
        super().__init__(MdGenderDatasetLoader.GROUP_NAME, MdGenderDatasetLoader.DATASET_NAME)

    def load(self):
        dataset = load_dataset("facebook/md_gender_bias", name="funpedia")
        all_examples = []
        female_count = 0
        for split in ("train", "validation", "test"):
            for text, entity, gender in zip(
                dataset[split]["text"], dataset[split]["title"], dataset[split]["gender"]
            ):
                if gender == 0:
                    # skip gender-neutral
                    continue
                gender = {1: "female", 2: "male"}[gender]
                if gender == "female":
                    female_count += 1
                all_examples.append((text, entity, gender))

        # Shuffle and go through examples again to balance the labels
        random.shuffle(all_examples)
        
        result = []
        male_count = 0
        for text, entity, context_label in all_examples:
            if context_label == "male":
                if male_count >= female_count:
                    continue
                male_count += 1

            questions = []
            answers = []
            paraphrases = random.sample(self.question_paraphrases, NUM_QA_PER_SAMPLE)
            for paraphrase in paraphrases:
                question_label = random.choice(["female", "male"])
                question = "# " + paraphrase.format(question_label)
                answer = YES_TOKEN if context_label == question_label else NO_TOKEN
                questions.append(question)
                answers.append(answer)

            context = f"{text}\n\nThis text is about {entity}."

            result.append(
                ContextQASample(
                    context=context, questions=questions, answers=answers
                )
            )
        return result


class AgNewsDatasetLoader(DatasetLoader):
    GROUP_NAME = "ag_news"
    DATASET_NAME = "ag_news"
    DATA_FILES_PATH = os.path.join(env_utils.DEFAULT_DATA_DIR, "ag_news")

    def __init__(self):
        super().__init__(AgNewsDatasetLoader.GROUP_NAME, AgNewsDatasetLoader.DATASET_NAME)

    def load(self):
        label_to_topic = {
            "1" : "World News",
            "2" : "Sports",
            "3" : "Business",
            "4" : "Science/Technology"
        }
        labels = set(label_to_topic.keys())
        examples = []
        with open(os.path.join(self.DATA_FILES_PATH, "ag_news.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                correct_label = row["Class Index"]

                title = row["Title"]
                description = row["Description"]

                context = f"{title}\n\n{description}"
                questions = []
                answers = []
                
                paraphrases = random.sample(self.question_paraphrases, NUM_QA_PER_SAMPLE)
                for paraphrase in paraphrases:
                    incorrect_label = random.choice(list(labels - {correct_label}))
                    question_label = random.choice((correct_label, incorrect_label))
                    question = "# " + paraphrase.format(label_to_topic[question_label])
                    answer = YES_TOKEN if question_label == correct_label else NO_TOKEN
                    questions.append(question)
                    answers.append(answer)

                examples.append(ContextQASample(
                    context=context, questions=questions, answers=answers
                ))
        return examples


class GeometryOfTruthDatasetLoader(DatasetLoader):
    GROUP_NAME = "geometry_of_truth"
    DATA_FILES_PATH = os.path.join(env_utils.DEFAULT_DATA_DIR, "gmt")

    DATASET_NAMES = [
        "sp_en_trans",
        # "neg_sp_en_trans",
        "cities",
        "neg_cities",
        "smaller_than",
        "larger_than",
        "common_claim_true_false",
        "companies_true_false",
        # "counterfact_true_false",
    ]

    def load(self):
        examples = []
        filename = self.name + ".csv"
        with open(os.path.join(self.DATA_FILES_PATH, filename)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions = []
                paraphrases = random.sample(self.question_paraphrases, NUM_QA_PER_SAMPLE)
                for paraphrase in paraphrases:
                    question = "# " + paraphrase
                    questions.append(question)
                answer = {"0": NO_TOKEN, "1": YES_TOKEN}[row["label"]]
                answers = [answer] * NUM_QA_PER_SAMPLE

                example = ContextQASample(
                    context=row["statement"], questions=questions, answers=answers
                )
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
    GROUP_NAME = "sst2"
    DATASET_NAME = "sst2"

    def __init__(self):
        super().__init__(SstDatasetLoader.GROUP_NAME, SstDatasetLoader.DATASET_NAME)

    def load(self):
        dataset = load_dataset("stanfordnlp/sst2")
        result = []
        for split in ("train", "validation"):
            for sentence, label in zip(
                dataset[split]["sentence"], dataset[split]["label"]
            ):
                context_label = {0: "negative", 1: "positive"}[label]
                questions = []
                answers = []
                paraphrases = {
                    label : random.sample(self.question_paraphrases[label], NUM_QA_PER_SAMPLE)
                    for label in ("positive", "negative")
                }
                for i in range(NUM_QA_PER_SAMPLE):
                    question_label = random.choice(["negative", "positive"])
                    question = "# " + paraphrases[question_label][i]
                    answer = YES_TOKEN if context_label == question_label else NO_TOKEN
                    questions.append(question)
                    answers.append(answer)

                result.append(
                    ContextQASample(
                        context=sentence.strip(), questions=questions, answers=answers
                    )
                )
        return result


# TODO (arnab): Remove some of the poor performing relations.
RELATION_FILES_ROOT = os.path.join(env_utils.DEFAULT_DATA_DIR, "relations")
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
            return []

        examples = []
        with open(file_path, "r") as f:
            data_dict = json.load(f)
            prompt_templates = data_dict["prompt_templates"]
            objects = [sample["object"] for sample in data_dict["samples"]]
            objects = set(objects)
            for sample in data_dict["samples"]:
                template = random.choice(prompt_templates) + " {}."
                questions = random.sample(self.question_paraphrases, NUM_QA_PER_SAMPLE)
                examples.append(
                    ContextQASample(
                        context=template.format(sample["subject"], sample["object"]),
                        questions=["# " + q for q in questions],
                        answers=[YES_TOKEN] * NUM_QA_PER_SAMPLE,
                    )
                )
                questions = random.sample(self.question_paraphrases, NUM_QA_PER_SAMPLE)
                false_obj = random.choice(list(objects - {sample["object"]}))
                examples.append(
                    ContextQASample(
                        context=template.format(sample["subject"], false_obj),
                        questions=["# " + q for q in questions],
                        answers=[NO_TOKEN] * NUM_QA_PER_SAMPLE,
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
    supported_datasets: dict[tuple[str, str], DatasetLoader] = {
        (dataset.group, dataset.name): dataset
        for dataset in (
            GeometryOfTruthDatasetLoader.get_all_loaders()
            + RelationDatasetLoader.get_all_loaders()
            + [
                SstDatasetLoader(),
                MdGenderDatasetLoader(),
                AgNewsDatasetLoader()
              ]
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
        for group, name in dataset_names:
            dataset = DatasetManager.supported_datasets[(group, name)]
            examples.extend(dataset.load())
        return DatasetManager(examples, batch_size, shuffle)

    @staticmethod
    def from_dataset_group(group, **kwargs):
        datasets = DatasetManager.list_datasets_by_group(group)
        names = datasets[group]

        return DatasetManager.from_named_datasets(
            zip([group] * len(names), names), **kwargs
        )

    @staticmethod
    def list_datasets_by_group(want_group=None):
        result = {}
        for group, name in DatasetManager.supported_datasets:
            if want_group is not None and group != want_group:
                continue
            if group not in result:
                result[group] = []
            result[group].append(name)
        return result
