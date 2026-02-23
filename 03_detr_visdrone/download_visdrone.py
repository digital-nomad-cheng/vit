"""
Download and extract the VisDrone-DET dataset (Task 1: Object Detection in Images).

Downloads trainset (~1.44 GB) and valset (~0.07 GB) from Google Drive,
extracts them, and organizes into the expected directory structure.

Usage:
    uv run python 03_detr_visdrone/download_visdrone.py
    uv run python 03_detr_visdrone/download_visdrone.py --data-dir ./my_data
"""

import argparse
import zipfile
from pathlib import Path

import gdown


DATASETS = {
    "VisDrone2019-DET-train": {
        "url": "https://drive.google.com/uc?id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
        "filename": "VisDrone2019-DET-train.zip",
    },
    "VisDrone2019-DET-val": {
        "url": "https://drive.google.com/uc?id=1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
        "filename": "VisDrone2019-DET-val.zip",
    },
}


def download_and_extract(url: str, filename: str, data_dir: Path) -> None:
    """Download a zip from Google Drive and extract it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / filename
    extracted_path = data_dir / zip_path.stem

    if extracted_path.is_dir():
        print(f"  ✓ Already extracted: {extracted_path}")
        return

    if not zip_path.is_file():
        print(f"  Downloading {filename} ...")
        gdown.download(url, str(zip_path), quiet=False)
    else:
        print(f"  ✓ Already downloaded: {zip_path}")

    print(f"  Extracting {filename} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"  ✓ Extracted to {extracted_path}")


def main():
    parser = argparse.ArgumentParser(description="Download VisDrone-DET dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to download and extract dataset into",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Which splits to download",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    print(f"Data directory: {data_dir}\n")

    for split in args.splits:
        key = f"VisDrone2019-DET-{split}"
        info = DATASETS[key]
        print(f"[{split}]")
        download_and_extract(info["url"], info["filename"], data_dir)
        print()

    print("Done! Dataset is ready.")
    print(f"  Train images: {data_dir / 'VisDrone2019-DET-train' / 'images'}")
    print(f"  Val images:   {data_dir / 'VisDrone2019-DET-val' / 'images'}")


if __name__ == "__main__":
    main()
