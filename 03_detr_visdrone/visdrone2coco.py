"""
Convert VisDrone-DET annotations to COCO JSON format.

VisDrone format (per line):
    bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion

Categories 0 (ignored regions) and 11 (others) are filtered out.
Categories 1-10 are re-mapped to 0-9 (zero-indexed).

Usage:
    uv run python 03_detr_visdrone/visdrone2coco.py
    uv run python 03_detr_visdrone/visdrone2coco.py --data-dir ./03_detr_visdrone/data
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import VISDRONE_CATEGORIES

# Mapping from VisDrone category ID → COCO category ID
# VisDrone: 0=ignored, 1=pedestrian, ..., 10=motor, 11=others
CATEGORY_MAP = {i: i - 1 for i in range(1, 11)}  # {1:0, 2:1, ..., 10:9}
IGNORED_CATEGORIES = {0, 11}


def convert_split(images_dir: str, annotations_dir: str, output_path: str) -> dict:
    """Convert one split of VisDrone annotations to COCO format."""
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    coco = {
        "images": [],
        "annotations": [],
        "categories": VISDRONE_CATEGORIES,
    }

    ann_id = 0
    image_files = sorted(images_dir.glob("*.jpg"))

    if not image_files:
        # Also check for png
        image_files = sorted(images_dir.glob("*.png"))

    print(f"  Found {len(image_files)} images in {images_dir}")

    for img_idx, img_path in enumerate(image_files):
        # Read image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        image_info = {
            "id": img_idx,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        }
        coco["images"].append(image_info)

        # Parse corresponding annotation file
        ann_path = annotations_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 8:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_width = float(parts[2])
                bbox_height = float(parts[3])
                # parts[4] = score (confidence, not used)
                category = int(parts[5])
                # parts[6] = truncation, parts[7] = occlusion

                # Skip ignored and "others" categories
                if category in IGNORED_CATEGORIES:
                    continue

                # Skip zero-area boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                coco_category = CATEGORY_MAP[category]

                annotation = {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": coco_category,
                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                }
                coco["annotations"].append(annotation)
                ann_id += 1

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)

    n_images = len(coco["images"])
    n_annotations = len(coco["annotations"])
    print(f"  Saved {n_images} images, {n_annotations} annotations → {output_path}")

    return coco


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone to COCO format")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Root data directory containing VisDrone2019-DET-* folders",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f"Data directory: {data_dir}\n")

    splits = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
    }

    for split_name, folder_name in splits.items():
        split_dir = os.path.join(data_dir, folder_name)
        images_dir = os.path.join(split_dir, "images")
        annotations_dir = os.path.join(split_dir, "annotations")
        output_path = os.path.join(data_dir, f"{split_name}.json")

        if not os.path.isdir(images_dir):
            print(f"[{split_name}] Skipping — {images_dir} not found")
            continue

        print(f"[{split_name}]")
        convert_split(images_dir, annotations_dir, output_path)
        print()

    print("COCO conversion complete!")


if __name__ == "__main__":
    main()
