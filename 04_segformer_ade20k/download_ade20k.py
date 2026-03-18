"""
Download ADE20K dataset (ADEChallengeData2016).

Downloads the ~923 MB zip file and extracts it into the data/ directory.

Usage:
    uv run python 04_segformer_ade20k/download_ade20k.py
    uv run python 04_segformer_ade20k/download_ade20k.py --data-dir ./my_data
"""

import argparse
import os
import zipfile

# The ADEChallengeData2016 download URL
ADE20K_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def download_ade20k(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "ADEChallengeData2016.zip")
    extract_dir = os.path.join(data_dir, "ADEChallengeData2016")

    if os.path.isdir(extract_dir):
        print(f"Dataset already exists at {extract_dir}")
        # Quick check
        train_dir = os.path.join(extract_dir, "images", "training")
        if os.path.isdir(train_dir):
            n = len([f for f in os.listdir(train_dir) if f.endswith(".jpg")])
            print(f"  Training images: {n}")
        return

    # Download
    if not os.path.isfile(zip_path):
        print(f"Downloading ADE20K to {zip_path} ...")
        print(f"  URL: {ADE20K_URL}")
        print("  This may take a few minutes (~923 MB)")

        try:
            import gdown
            gdown.download(ADE20K_URL, zip_path, quiet=False)
        except ImportError:
            import urllib.request
            urllib.request.urlretrieve(ADE20K_URL, zip_path)
        print("Download complete.")
    else:
        print(f"Zip already downloaded: {zip_path}")

    # Extract
    print(f"Extracting to {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print("Extraction complete.")

    # Verify
    train_imgs = os.path.join(extract_dir, "images", "training")
    val_imgs = os.path.join(extract_dir, "images", "validation")
    n_train = len(os.listdir(train_imgs)) if os.path.isdir(train_imgs) else 0
    n_val = len(os.listdir(val_imgs)) if os.path.isdir(val_imgs) else 0
    print(f"ADE20K ready: {n_train} train, {n_val} val images")

    # Optionally remove zip
    # os.remove(zip_path)


def main():
    parser = argparse.ArgumentParser(description="Download ADE20K dataset")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory to download dataset into",
    )
    args = parser.parse_args()
    download_ade20k(args.data_dir)


if __name__ == "__main__":
    main()
