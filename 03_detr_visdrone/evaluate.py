"""
Evaluate DETR on VisDrone-DET using COCO mAP metrics.

Usage:
    uv run python 03_detr_visdrone/evaluate.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth
    uv run python 03_detr_visdrone/evaluate.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth --score-thresh 0.5
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_dataloaders
from detr import build_detr
from losses import box_cxcywh_to_xyxy
from config import NUM_CLASSES, CATEGORY_NAMES



@torch.no_grad()
def run_evaluation(
    model,
    val_loader,
    device: torch.device,
    score_thresh: float = 0.0,
) -> list[dict]:
    """Run model on val set and collect predictions in COCO results format."""
    model.eval()
    results = []

    for images, masks, targets in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images, masks)
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        pred_boxes = outputs["pred_boxes"]    # (B, Q, 4) cxcywh [0,1]

        # Convert to probabilities (exclude no-object class)
        probs = pred_logits.softmax(-1)[:, :, :-1]  # (B, Q, C)

        for b in range(images.shape[0]):
            image_id = targets[b]["image_id"].item()
            orig_h, orig_w = targets[b]["orig_size"].tolist()

            scores, labels = probs[b].max(-1)  # (Q,), (Q,)
            boxes = pred_boxes[b]               # (Q, 4) cxcywh normalized

            # Filter by score
            keep = scores > score_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # Convert to xyxy absolute coords
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            boxes_xyxy[:, 0::2] *= orig_w
            boxes_xyxy[:, 1::2] *= orig_h

            # Convert to xywh for COCO
            boxes_xywh = boxes_xyxy.clone()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]

            for i in range(len(scores)):
                results.append({
                    "image_id": image_id,
                    "category_id": labels[i].item(),
                    "bbox": boxes_xywh[i].cpu().tolist(),
                    "score": scores[i].item(),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DETR on VisDrone-DET")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Path to VisDrone data directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--score-thresh", type=float, default=0.01,
                        help="Score threshold for filtering predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Build model
    num_queries = ckpt_args.get("num_queries", 100)
    model = build_detr(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,  # weights come from checkpoint
        num_queries=num_queries,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # Build val dataloader
    _, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_size=args.max_size,
    )
    print(f"Val batches: {len(val_loader)}")

    # Run evaluation
    print("Running inference on validation set...")
    results = run_evaluation(model, val_loader, device, args.score_thresh)
    print(f"Collected {len(results)} predictions")

    if not results:
        print("No predictions — cannot compute mAP.")
        return

    # Save results
    results_path = os.path.join(os.path.dirname(args.checkpoint), "val_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {results_path}")

    # COCO evaluation
    ann_file = os.path.join(args.data_dir, "val.json")
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print per-category AP
    print("\nPer-category AP@[.50:.95]:")
    print("-" * 40)
    for cat_id, cat_name in enumerate(CATEGORY_NAMES):
        coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        ap = coco_eval_cat.stats[0]  # AP@[.50:.95]
        print(f"  {cat_name:20s}: {ap:.4f}")


if __name__ == "__main__":
    main()
