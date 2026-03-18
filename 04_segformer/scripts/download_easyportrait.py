"""
Download EasyPortrait dataset (v1 or v2).

EasyPortrait is a face parsing & portrait segmentation dataset with 9 classes:
  0: Background, 1: Person, 2: Face skin, 3: Left brow, 4: Right brow,
  5: Left eye, 6: Right eye, 7: Lips, 8: Teeth

  v1: 20,000 images (14k train / 2k val / 4k test), ~26 GB
  v2: 40,000 images (30k train / 4k val / 6k test), ~92 GB

Downloads images.zip and annotations.zip from the official mirrors and
extracts them into the data directory.

Usage:
    uv run python 04_segformer_ade20k/scripts/download_easyportrait.py
    uv run python 04_segformer_ade20k/scripts/download_easyportrait.py --version v1
    uv run python 04_segformer_ade20k/scripts/download_easyportrait.py --data-dir ./my_data --version v2

References:
    https://github.com/hukenovs/easyportrait
    https://www.kaggle.com/datasets/kapitanov/easyportrait
"""

import argparse
import os
import sys
import urllib.request
import zipfile

# Direct download URLs (no Kaggle API needed)
URLS = {
    "v1": {
        "images": "https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/images.zip",
        "annotations": "https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/annotations.zip",
    },
    "v2": {
        "images": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/v2/images.zip",
        "annotations": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/v2/annotations.zip",
    },
}

# Expected image counts per split
EXPECTED_COUNTS = {
    "v1": {"train": 14_000, "val": 2_000, "test": 4_000},
    "v2": {"train": 30_000, "val": 4_000, "test": 6_000},
}


class _DownloadProgressBar:
    """Simple progress reporter for urllib downloads."""

    def __init__(self, filename: str):
        self.filename = filename
        self.last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(int(downloaded * 100 / total_size), 100)
        if pct != self.last_pct:
            self.last_pct = pct
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  {self.filename}: {mb:.0f}/{total_mb:.0f} MB ({pct}%)"
            )
            sys.stdout.flush()
            if pct == 100:
                sys.stdout.write("\n")


def _download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    filename = os.path.basename(dest)
    print(f"Downloading {filename} ...")
    print(f"  URL: {url}")
    try:
        import gdown
        gdown.download(url, dest, quiet=False)
    except ImportError:
        urllib.request.urlretrieve(url, dest, reporthook=_DownloadProgressBar(filename))
    print(f"  Saved to {dest}")


def _extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract a zip file."""
    print(f"Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("  Done.")


def _count_images(directory: str) -> int:
    """Count image files in a directory."""
    if not os.path.isdir(directory):
        return 0
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def download_easyportrait(data_dir: str, version: str = "v2") -> None:
    """Download and extract the EasyPortrait dataset.

    Args:
        data_dir: Root directory for the dataset (e.g. data/EasyPortrait).
        version: Dataset version, "v1" or "v2".
    """
    assert version in URLS, f"Unknown version '{version}', choose from: {list(URLS)}"
    urls = URLS[version]

    os.makedirs(data_dir, exist_ok=True)

    images_dir = os.path.join(data_dir, "images")
    annotations_dir = os.path.join(data_dir, "annotations")

    # Check if already extracted
    if os.path.isdir(images_dir) and os.path.isdir(annotations_dir):
        n_train = _count_images(os.path.join(images_dir, "train"))
        n_val = _count_images(os.path.join(images_dir, "val"))
        n_test = _count_images(os.path.join(images_dir, "test"))
        if n_train > 0:
            print(f"EasyPortrait {version} already exists at {data_dir}")
            print(f"  train: {n_train}, val: {n_val}, test: {n_test}")
            return

    # Download and extract images
    images_zip = os.path.join(data_dir, "images.zip")
    if not os.path.isfile(images_zip):
        _download_file(urls["images"], images_zip)
    else:
        print(f"images.zip already downloaded: {images_zip}")

    if not os.path.isdir(images_dir):
        _extract_zip(images_zip, data_dir)

    # Download and extract annotations
    annotations_zip = os.path.join(data_dir, "annotations.zip")
    if not os.path.isfile(annotations_zip):
        _download_file(urls["annotations"], annotations_zip)
    else:
        print(f"annotations.zip already downloaded: {annotations_zip}")

    if not os.path.isdir(annotations_dir):
        _extract_zip(annotations_zip, data_dir)

    # Verify counts
    expected = EXPECTED_COUNTS[version]
    for split in ("train", "val", "test"):
        n_imgs = _count_images(os.path.join(images_dir, split))
        n_masks = _count_images(os.path.join(annotations_dir, split))
        status = "✓" if n_imgs >= expected[split] else "✗"
        print(f"  {status} {split}: {n_imgs} images, {n_masks} masks"
              f" (expected ~{expected[split]})")

    print(f"\nEasyPortrait {version} ready at: {data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download EasyPortrait dataset"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "EasyPortrait"),
        help="Directory to download dataset into (default: data/EasyPortrait)",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="Dataset version (default: v2). v1=20k images, v2=40k images",
    )
    args = parser.parse_args()
    download_easyportrait(args.data_dir, version=args.version)


if __name__ == "__main__":
    main()
