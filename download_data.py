"""Download, validate, RGB-normalize and split the official Cats vs Dogs dataset."""
from __future__ import annotations

import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
EXTRACT_DIR = RAW_DIR / "PetImages"
ZIP_PATH = RAW_DIR / "kagglecatsanddogs_5340.zip"
DATA_DIR = ROOT / "data"
DATA_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
SEED = 42
RGB_NORMALIZATION_VERSION = "2026-11-21"


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        print(f"Using existing archive: {ZIP_PATH}")
        return

    print("Downloading dataset (about 787 MiB). This may take a while...")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 / total_size)
            print(f"\rDownload progress: {percent:5.1f}%", end="")

    urllib.request.urlretrieve(DATA_URL, ZIP_PATH, reporthook=report)
    print("\nDownload complete.")


def extract_dataset() -> None:
    if (EXTRACT_DIR / "Cat").is_dir() and (EXTRACT_DIR / "Dog").is_dir():
        print("Dataset already extracted.")
        return
    print("Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(RAW_DIR)
    if not (EXTRACT_DIR / "Cat").is_dir() or not (EXTRACT_DIR / "Dog").is_dir():
        raise FileNotFoundError("Expected PetImages/Cat and PetImages/Dog folders in archive.")


def is_valid_image(path: Path) -> bool:
    """Return True only if Pillow can fully decode and convert the image to RGB."""
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            rgb.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return False


def write_rgb_jpeg(source: Path, destination: Path) -> bool:
    """Re-encode every accepted source as a standard 3-channel RGB JPEG.

    This avoids TensorFlow decode errors from unusual source files (for example,
    images whose internal channel count is 2) that Pillow can still convert.
    """
    try:
        with Image.open(source) as image:
            image.load()
            rgb = image.convert("RGB")
            rgb.save(destination, format="JPEG", quality=95, optimize=True)
        return True
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        print(f"Skipping unreadable image: {source.name} ({exc})")
        return False


def make_splits() -> None:
    rng = random.Random(SEED)
    split_names = ("train", "val", "test")

    # Clear old split files so the RGB-normalized images replace the originals.
    for split in split_names:
        for class_name in ("cats", "dogs"):
            target = DATA_DIR / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for old_file in target.iterdir():
                if old_file.is_file():
                    old_file.unlink()

    total_written = 0
    for source_class, target_class in (("Cat", "cats"), ("Dog", "dogs")):
        source_dir = EXTRACT_DIR / source_class
        candidates = sorted(source_dir.glob("*.jpg"))
        valid_files = [path for path in candidates if is_valid_image(path)]
        rng.shuffle(valid_files)
        n = len(valid_files)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        groups = {
            "train": valid_files[:n_train],
            "val": valid_files[n_train:n_train + n_val],
            "test": valid_files[n_train + n_val:],
        }

        for split, files in groups.items():
            destination = DATA_DIR / split / target_class
            written = 0
            for index, src in enumerate(files):
                dest = destination / f"{target_class}_{index:05d}.jpg"
                if write_rgb_jpeg(src, dest):
                    written += 1
            print(f"{target_class:4s} | {split:5s}: {written:5d} RGB images")
            total_written += written

    print(f"\nDataset ready. RGB images written: {total_written}")
    print("All training images have been re-saved as standard 3-channel RGB JPEGs.")
    print(f"RGB_NORMALIZATION_VERSION={RGB_NORMALIZATION_VERSION}")
    print("Split: approximately 80% train, 10% validation, 10% test.")


def main() -> None:
    download_dataset()
    extract_dataset()
    make_splits()
    print("\nDataset is ready. Next step: python train.py --batch-size 16")


if __name__ == "__main__":
    main()
