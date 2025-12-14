import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_logger

logger = get_logger("preprocess")

BASE_RAW_DIR = "/data/raw"
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_DIRS = {"consensus", "sample"}

subdirs = [
    d for d in os.listdir(BASE_RAW_DIR)
    if os.path.isdir(os.path.join(BASE_RAW_DIR, d))
]

if len(subdirs) == 1 and subdirs[0] not in IGNORE_DIRS:
    DATA_ROOT = os.path.join(BASE_RAW_DIR, subdirs[0])
    logger.info(f"Detected ZIP internal root directory: {DATA_ROOT}")
else:
    DATA_ROOT = BASE_RAW_DIR
    logger.info(f"Using raw directory directly: {DATA_ROOT}")


def extract_label_from_choice(choice_str: str) -> int:
    first_part = choice_str.split("-")[0].strip()
    numeric = first_part.split(" ")[0].strip()
    return int(numeric)


def process_one_json_file(path):
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.warning(f"Invalid JSON file, skipped: {path}")
        return examples

    if not isinstance(data, list):
        logger.info(f"Not a Label Studio JSON file, skipped: {path}")
        return examples

    for task in data:
        text = task.get("data", {}).get("text", "").strip()
        if not text:
            continue

        anns = task.get("annotations", [])
        if not anns:
            continue

        results = anns[0].get("result", [])
        if not results:
            continue

        choices = results[0].get("value", {}).get("choices", [])
        if not choices:
            continue

        try:
            label = extract_label_from_choice(choices[0])
        except Exception:
            continue

        examples.append((text, label))

    return examples


def collect_all_labeled_examples():
    all_examples = []

    for annotator_name in os.listdir(DATA_ROOT):
        if annotator_name in IGNORE_DIRS:
            logger.info(f"Ignored directory: {annotator_name}")
            continue

        full_dir = os.path.join(DATA_ROOT, annotator_name)
        if not os.path.isdir(full_dir):
            continue

        logger.info(f"Processing annotator: {annotator_name}")

        for fname in os.listdir(full_dir):
            if fname.lower().endswith(".json"):
                all_examples.extend(
                    process_one_json_file(os.path.join(full_dir, fname))
                )

    return all_examples


def main():
    logger.info("Loading labeled data...")

    examples = collect_all_labeled_examples()
    logger.info(f"Total labeled examples collected: {len(examples)}")

    if len(examples) == 0:
        raise RuntimeError("NO LABELED DATA FOUND. Please check the dataset ZIP.")

    df = pd.DataFrame(examples, columns=["text", "label"])

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding="utf-8-sig")

    logger.info("train.csv, val.csv and test.csv successfully created.")


if __name__ == "__main__":
    main()
