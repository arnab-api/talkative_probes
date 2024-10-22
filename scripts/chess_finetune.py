import argparse
import logging
import os
import time
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from src.functional import get_module_nnsight
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils
import shutil
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


class PGNDataset(torch.utils.data.Dataset):
    def __init__(self, pgn_ds, tokenizer):
        self.pgn_ds = pgn_ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pgn_ds)

    def __getitem__(self, idx):
        item = self.pgn_ds[idx]
        text = item["transcript"]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, pgn_ds, tokenizer):
        self.pgn_ds = pgn_ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pgn_ds)

    def __getitem__(self, idx):
        item = self.pgn_ds[idx]
        text = item["text"]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def chess_finetune(
    model_key: str,
    limit_training_steps: int = 100000,
):
    # loading the model
    mt = ModelandTokenizer(
        model_key=model_key,
        torch_dtype=torch.float32,
    )

    # loading the datasets
    pgn_ds = load_dataset(
        "adamkarvonen/chess_games", data_files="lichess_6gb.zip", streaming=False
    )
    pgn_ds = pgn_ds["train"].train_test_split(test_size=0.1)

    train_dataset = PGNDataset(pgn_ds["train"], tokenizer=mt.tokenizer)
    test_dataset = PGNDataset(pgn_ds["test"], tokenizer=mt.tokenizer)

    wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    wiki_ds = wiki_ds["train"].train_test_split(test_size=0.1)

    train_wiki = WikiDataset(wiki_ds["train"], tokenizer=mt.tokenizer)
    test_wiki = WikiDataset(wiki_ds["test"], tokenizer=mt.tokenizer)

    #################### Hyperparameters ####################
    learning_rate = 5e-5
    batch_size = 6

    model_save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, "chess_model_finetuned"
    )
    os.makedirs(model_save_dir, exist_ok=True)
    wandb_log_interval = 50
    checkpoint_interval = 1000
    num_warmup_steps = min(1000, limit_training_steps // 10)
    limit_training_steps = limit_training_steps
    ##########################################################

    model = mt._model
    model.train()
    device = mt.device

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_wiki_loader = DataLoader(train_wiki, batch_size=batch_size // 2, shuffle=True)
    test_wiki_loader = DataLoader(test_wiki, batch_size=batch_size // 2, shuffle=False)

    limit_training_steps = min(
        limit_training_steps, len(train_loader), len(train_wiki_loader)
    )

    logger.info(
        f"Limit training steps: {limit_training_steps} | {len(train_loader)=} | {len(train_wiki_loader)=}"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=limit_training_steps,
    )

    # wandb
    wandb.init(
        entity="dl-homeworks",
        project="talkative_probes",
        name=f"{model_key}_finetune",
        config={
            "model_key": model_key,
            "learning_rate": learning_rate,
            "wandb_log_interval": wandb_log_interval,
            "checkpoint_interval": checkpoint_interval,
            "num_warmup_steps": num_warmup_steps,
            "batch_size": batch_size,
        },
    )

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pgn_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        chess_loss = pgn_outputs.loss

        wiki_batch = next(iter(train_wiki_loader))
        wiki_input_ids = wiki_batch["input_ids"].to(device)
        wiki_attention_mask = wiki_batch["attention_mask"].to(device)
        wiki_labels = wiki_batch["labels"].to(device)

        wiki_outputs = model(
            input_ids=wiki_input_ids,
            attention_mask=wiki_attention_mask,
            labels=wiki_labels,
        )
        wiki_loss = wiki_outputs.loss

        loss = chess_loss + wiki_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % wandb_log_interval == 0:
            log_report = {
                "loss": loss.item(),
                "chess_loss": chess_loss.item(),
                "wiki_loss": wiki_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            logger.info(f"Step {step + 1}: {log_report}")
            wandb.log(log_report)

        if ((step + 1) % checkpoint_interval == 0) or (
            step + 1
        ) == limit_training_steps:
            if len(os.listdir(model_save_dir)) > 0:
                last_checkpoint_path = os.path.join(
                    model_save_dir, os.listdir(model_save_dir)[-1]
                )
                remove_dir(last_checkpoint_path)

            new_checkpoint_path = os.path.join(model_save_dir, f"checkpoint-{step + 1}")
            model.save_pretrained(new_checkpoint_path)

            if step + 1 == limit_training_steps:
                logger.info("Finetuning complete.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model key to use for fine-tuning.",
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--limit_training_steps",
        type=int,
        default=100000,
        help="The maximum number of training steps.",
    )
    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info("args")

    chess_finetune(
        model_key=args.model,
        limit_training_steps=args.limit_training_steps,
    )
