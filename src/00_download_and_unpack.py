import os
import requests
import zipfile
from utils import get_logger

logger = get_logger("download")

ZIP_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"

DATA_ROOT = "/data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")


def download_zip(zip_path: str):
    logger.info("Downloading dataset...")
    with requests.get(ZIP_URL, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logger.info(f"ZIP file downloaded to: {zip_path}")


def unpack_zip(zip_path: str, target_dir: str):
    logger.info("Extracting ZIP archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    logger.info(f"Archive extracted to: {target_dir}")


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    raw_has_content = any(os.scandir(RAW_DIR))
    if raw_has_content:
        logger.info("/data/raw is not empty, skipping download and extraction.")
        return

    zip_path = os.path.join(DATA_ROOT, "dataset.zip")
    download_zip(zip_path)
    unpack_zip(zip_path, RAW_DIR)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    logger.info("Temporary ZIP removed. Download and extraction finished.")


if __name__ == "__main__":
    main()
