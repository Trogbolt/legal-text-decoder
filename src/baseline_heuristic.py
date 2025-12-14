import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import get_logger
import config


def word_count(text: str) -> int:
    return len(text.split())


def main():
    logger = get_logger("heuristic")
    logger.info("=== HEURISTIC BASELINE (WORD COUNT) START ===")

    train_df = pd.read_csv(config.PATH_TRAIN, encoding="utf-8-sig")
    test_df = pd.read_csv(config.PATH_TEST, encoding="utf-8-sig")

    train_df["wc"] = train_df["text"].apply(word_count)
    test_df["wc"] = test_df["text"].apply(word_count)

    q20, q40, q60, q80 = np.percentile(train_df["wc"], [20, 40, 60, 80])
    logger.info(f"Thresholds: {q20}, {q40}, {q60}, {q80}")

    def predict(wc):
        if wc <= q20:
            return 5
        elif wc <= q40:
            return 4
        elif wc <= q60:
            return 3
        elif wc <= q80:
            return 2
        else:
            return 1

    preds = test_df["wc"].apply(predict)
    y_true = test_df["label"]

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="weighted")

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-weighted: {f1:.4f}")
    logger.info("\n%s", classification_report(y_true, preds, zero_division=0))

    os.makedirs("/app/output/metrics", exist_ok=True)
    with open("/app/output/metrics/heuristic.json", "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

    logger.info("=== HEURISTIC BASELINE FINISHED ===")


if __name__ == "__main__":
    main()
