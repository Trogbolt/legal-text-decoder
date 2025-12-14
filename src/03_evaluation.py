import os
import json
import torch
import pandas as pd
import warnings

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import config
from utils import get_logger

warnings.filterwarnings(
    "ignore",
    message=".*encoder_attention_mask.*",
    category=FutureWarning,
)


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


@torch.no_grad()
def main():
    logger = get_logger("evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("=== TEST EVALUATION STARTED ===")

    test_df = pd.read_csv(config.PATH_TEST, encoding="utf-8-sig")
    labels = test_df["label"].to_numpy(dtype=int) - 1

    model_dir = config.MODEL_BEST_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    loader = DataLoader(
        TextDataset(test_df["text"].tolist(), labels, tokenizer, config.MAX_LENGTH),
        batch_size=config.EVAL_BATCH_SIZE,
    )

    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labs = batch["labels"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info(f"Test F1-weighted: {f1:.4f}")

    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2, 3, 4],
        target_names=["1", "2", "3", "4", "5"],
        zero_division=0,
    )
    logger.info("\n%s", report)

    cm = confusion_matrix(all_labels, all_preds)
    logger.info("Confusion matrix:")
    logger.info("\n%s", cm)

    os.makedirs("/app/output/metrics", exist_ok=True)
    with open("/app/output/metrics/bert_test.json", "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

    logger.info("=== TEST EVALUATION FINISHED ===")


if __name__ == "__main__":
    main()
