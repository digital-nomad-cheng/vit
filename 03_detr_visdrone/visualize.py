"""
Visualize DETR predictions on images.

Draws bounding boxes with class labels and confidence scores.

Usage:
    uv run python 03_detr_visdrone/visualize.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth
    uv run python 03_detr_visdrone/visualize.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth --images img1.jpg img2.jpg
    uv run python 03_detr_visdrone/visualize.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth --num-images 10 --score-thresh 0.5
"""

import argparse
import os
import random
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from detr import build_detr
from losses import box_cxcywh_to_xyxy
from config import NUM_CLASSES, CATEGORY_NAMES, IMAGENET_MEAN, IMAGENET_STD

# Distinct colors for each class
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]

def preprocess(img: Image.Image, max_size: int = 800) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess a single image for DETR inference."""
    w, h = img.size
    scale = min(max_size / max(h, w), 1.0)
    new_h, new_w = int(h * scale), int(w * scale)

    transform = T.Compose([
        T.Resize((new_h, new_w)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    mask = torch.zeros(1, new_h, new_w, dtype=torch.bool)
    return img_tensor, mask


@torch.no_grad()
def predict(
    model,
    img: Image.Image,
    device: torch.device,
    max_size: int = 800,
    score_thresh: float = 0.5,
) -> list[dict]:
    """Run DETR on a single image and return detections."""
    model.eval()
    orig_w, orig_h = img.size

    img_tensor, mask = preprocess(img, max_size)
    img_tensor = img_tensor.to(device)
    mask = mask.to(device)

    outputs = model(img_tensor, mask)

    # Post-process
    probs = outputs["pred_logits"].softmax(-1)[0, :, :-1]  # (Q, C)
    boxes = outputs["pred_boxes"][0]  # (Q, 4) cxcywh normalized

    scores, labels = probs.max(-1)
    keep = scores > score_thresh

    scores = scores[keep].cpu()
    labels = labels[keep].cpu()
    boxes = boxes[keep].cpu()

    # Convert to absolute xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes_xyxy[:, 0::2] *= orig_w
    boxes_xyxy[:, 1::2] *= orig_h

    detections = []
    for i in range(len(scores)):
        detections.append({
            "box": boxes_xyxy[i].tolist(),
            "label": labels[i].item(),
            "score": scores[i].item(),
        })

    return detections


def draw_detections(img: Image.Image, detections: list[dict],
                    line_width: int = 2) -> Image.Image:
    """Draw bounding boxes and labels on an image."""
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]

        color = COLORS[label % len(COLORS)]
        name = CATEGORY_NAMES[label]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background
        text = f"{name} {score:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), text, fill="white", font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize DETR predictions")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Specific image paths to visualize")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Data directory (used if --images not specified)")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "vis_output"),
                        help="Output directory for visualizations")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of random val images to visualize")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="Confidence threshold for predictions")
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    model = build_detr(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        num_queries=ckpt_args.get("num_queries", 100),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # Get image paths
    if args.images:
        image_paths = args.images
    else:
        val_dir = os.path.join(args.data_dir, "VisDrone2019-DET-val", "images")
        all_images = sorted([
            os.path.join(val_dir, f) for f in os.listdir(val_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])
        image_paths = random.sample(all_images, min(args.num_images, len(all_images)))

    # Run and visualize
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        img = Image.open(img_path).convert("RGB")

        detections = predict(model, img, device, args.max_size, args.score_thresh)
        print(f"  Found {len(detections)} objects")

        for det in detections:
            name = CATEGORY_NAMES[det['label']]
            print(f"    {name}: {det['score']:.3f} at [{det['box'][0]:.0f}, "
                  f"{det['box'][1]:.0f}, {det['box'][2]:.0f}, {det['box'][3]:.0f}]")

        annotated = draw_detections(img.copy(), detections)
        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        annotated.save(out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone! Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
