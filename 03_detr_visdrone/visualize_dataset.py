"""
Visualize VisDrone-DET dataset images with ground-truth bounding boxes.

Draws GT boxes color-coded by category with class name labels.

Usage:
    uv run python 03_detr_visdrone/visualize_dataset.py
    uv run python 03_detr_visdrone/visualize_dataset.py --split train --num-images 10
    uv run python 03_detr_visdrone/visualize_dataset.py --images path/to/img1.jpg path/to/img2.jpg
"""

import argparse
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

sys.path.insert(0, os.path.dirname(__file__))
from config import CATEGORY_NAMES

# Distinct colors for each class
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


def draw_gt_boxes(img: Image.Image, annotations: list[dict],
                  line_width: int = 2) -> Image.Image:
    """Draw ground-truth bounding boxes and labels on an image."""
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for ann in annotations:
        cat_id = ann["category_id"]
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h

        color = COLORS[cat_id % len(COLORS)]
        name = CATEGORY_NAMES[cat_id]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background
        text = name
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle(
            [x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color
        )
        draw.text((x1 + 2, y1 - text_h - 2), text, fill="white", font=font)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VisDrone-DET dataset with GT labels"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Data directory with COCO JSON annotations",
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val"],
        help="Which split to visualize",
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="Specific image filenames to visualize (basenames only)",
    )
    parser.add_argument(
        "--num-images", type=int, default=5,
        help="Number of random images to visualize",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "vis_dataset"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load COCO annotations
    ann_file = os.path.join(args.data_dir, f"{args.split}.json")
    print(f"Loading annotations: {ann_file}")
    coco = COCO(ann_file)

    # Determine which images to visualize
    if args.images:
        # Find image IDs by filename
        all_imgs = coco.imgs
        name_to_id = {v["file_name"]: k for k, v in all_imgs.items()}
        img_ids = []
        for name in args.images:
            if name in name_to_id:
                img_ids.append(name_to_id[name])
            else:
                print(f"  ⚠ Image not found in annotations: {name}")
    else:
        all_ids = list(coco.imgs.keys())
        if args.seed is not None:
            random.seed(args.seed)
        img_ids = random.sample(all_ids, min(args.num_images, len(all_ids)))

    # Get images directory
    split_folder = (
        "VisDrone2019-DET-train" if args.split == "train"
        else "VisDrone2019-DET-val"
    )
    images_dir = os.path.join(args.data_dir, split_folder, "images")

    print(f"Visualizing {len(img_ids)} images from {args.split} split\n")

    # Legend
    print("Category legend:")
    for i, name in enumerate(CATEGORY_NAMES):
        print(f"  {COLORS[i]}  {name}")
    print()

    for img_id in img_ids:
        img_info = coco.imgs[img_id]
        img_path = os.path.join(images_dir, img_info["file_name"])

        if not os.path.isfile(img_path):
            print(f"  ⚠ Image file not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        print(f"{img_info['file_name']}: {len(annotations)} objects")

        # Count per category
        cat_counts: dict[str, int] = {}
        for ann in annotations:
            name = CATEGORY_NAMES[ann["category_id"]]
            cat_counts[name] = cat_counts.get(name, 0) + 1
        for name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")

        annotated = draw_gt_boxes(img.copy(), annotations)
        out_name = os.path.splitext(img_info["file_name"])[0] + "_gt.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        annotated.save(out_path, quality=95)
        print(f"    → {out_path}\n")

    print(f"Done! Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
