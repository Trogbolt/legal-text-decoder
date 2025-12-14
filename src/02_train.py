import os
import time
import json
import torch
import pandas as pd
import warnings

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score, f1_score

import config
from utils import get_logger

# --- silence HF warning ---
warnings.filterwarnings(
    "ignore",
    message=".*encoder_attention_mask.*",
    category=FutureWarning,
)

# =============================================================================
# Dataset
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =============================================================================
# Utils
# =============================================================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="weighted"),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="weighted"),
    )


# =============================================================================
# Main
# =============================================================================
def main():
    logger = get_logger("train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("=== TRAINING STARTED ===")

    # --- Configuration ---
    logger.info("Configuration:")
    logger.info(f"MODEL_NAME       = {config.MODEL_NAME}")
    logger.info(f"EPOCHS           = {config.EPOCHS}")
    logger.info(f"TRAIN_BATCH_SIZE = {config.TRAIN_BATCH_SIZE}")
    logger.info(f"EVAL_BATCH_SIZE  = {config.EVAL_BATCH_SIZE}")
    logger.info(f"LEARNING_RATE    = {config.LEARNING_RATE}")
    logger.info(f"MAX_LENGTH       = {config.MAX_LENGTH}")
    logger.info(f"EARLY_STOPPING   = {config.EARLY_STOPPING_PATIENCE}")

    # --- Data ---
    train_df = pd.read_csv(config.PATH_TRAIN, encoding="utf-8-sig")
    val_df = pd.read_csv(config.PATH_VAL, encoding="utf-8-sig")

    train_labels = train_df["label"].to_numpy(dtype=int) - 1
    val_labels = val_df["label"].to_numpy(dtype=int) - 1

    # --- Model ---
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=5
    ).to(device)

    total, trainable, frozen = count_parameters(model)
    logger.info(
        f"Model parameters: total={total:,}, trainable={trainable:,}, frozen={frozen:,}"
    )

    # --- Loaders ---
    train_loader = DataLoader(
        TextDataset(train_df["text"].tolist(), train_labels, tokenizer, config.MAX_LENGTH),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TextDataset(val_df["text"].tolist(), val_labels, tokenizer, config.MAX_LENGTH),
        batch_size=config.EVAL_BATCH_SIZE,
    )

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    best_val_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, config.EPOCHS + 1):
        logger.info(f"--- Epoch {epoch}/{config.EPOCHS} ---")

        start = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        logger.info(
            f"Train | loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}"
        )
        logger.info(f"Epoch time: {(time.time()-start)/60:.2f} min")

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        logger.info(
            f"Val   | loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_dir = config.MODEL_BEST_DIR
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info("New BEST model saved (val_f1={:.4f})".format(best_val_f1))
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement ({epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE})"
            )
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered")
                break

    # --- Save last model ---
    last_dir = config.MODEL_LAST_DIR
    os.makedirs(last_dir, exist_ok=True)
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)

    os.makedirs(last_dir, exist_ok=True)
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
    logger.info(f"Last model saved to {last_dir}")

    # --- Save metrics ---
    os.makedirs("/app/output/metrics", exist_ok=True)
    with open("/app/output/metrics/bert_train.json", "w") as f:
        json.dump({"best_val_f1": best_val_f1}, f, indent=2)

    logger.info("=== TRAINING FINISHED ===")


if __name__ == "__main__":
    main()
