import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import get_logger
import config


def main():
    logger = get_logger("baseline")
    logger.info("=== TF-IDF BASELINE START ===")

    train_df = pd.read_csv(config.PATH_TRAIN, encoding="utf-8-sig")
    test_df = pd.read_csv(config.PATH_TEST, encoding="utf-8-sig")

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, train_df["label"])

    preds = clf.predict(X_test)

    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds, average="weighted")

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-weighted: {f1:.4f}")
    logger.info("\n%s", classification_report(test_df["label"], preds, zero_division=0))

    os.makedirs("/app/output/metrics", exist_ok=True)
    with open("/app/output/metrics/tfidf.json", "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

    logger.info("=== TF-IDF BASELINE FINISHED ===")


if __name__ == "__main__":
    main()
