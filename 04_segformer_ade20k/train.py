"""
Train SegFormer-B0 on ADE20K semantic segmentation.

Usage:
    uv run python 04_segformer_ade20k/train.py
    uv run python 04_segformer_ade20k/train.py --pretrained --epochs 50
    uv run python 04_segformer_ade20k/train.py --batch-size 8 --lr 6e-5

Prerequisites:
    1. Download ADE20K: uv run python 04_segformer_ade20k/scripts/download_ade20k.py
    2. (Optional) Install huggingface_hub for pretrained weights:
       uv pip install huggingface_hub
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from segformer import segformer_b0, load_pretrained_segformer_b0
from dataset.ade20k import build_dataloaders, NUM_CLASSES, IGNORE_INDEX


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_miou(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX) -> tuple[float, list[float]]:
    """Compute mean Intersection over Union (mIoU).

    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) ground truth class indices
    Returns:
        miou: scalar mean IoU
        per_class_iou: list of per-class IoU values
    """
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    per_class_iou = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            per_class_iou.append(intersection / union)
        # Skip classes not present in target — don't count as 0

    miou = sum(per_class_iou) / max(len(per_class_iou), 1)
    return miou, per_class_iou


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum: int = 1,
    print_freq: int = 50,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, miou)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    t0 = time.time()

    optimizer.zero_grad()

    for step, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward: model outputs at 1/4 resolution
        logits = model(images)  # (B, C, H/4, W/4)

        # Upsample logits to match mask resolution
        logits = F.interpolate(logits, size=masks.shape[1:],
                               mode='bilinear', align_corners=False)

        loss = criterion(logits, masks)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item() * grad_accum * images.size(0)
        valid = masks != IGNORE_INDEX
        preds = logits.argmax(1)
        total_correct += (preds[valid] == masks[valid]).sum().item()
        total_pixels += valid.sum().item()

        if (step + 1) % print_freq == 0:
            avg_loss = total_loss / ((step + 1) * images.size(0))
            pixel_acc = total_correct / max(total_pixels, 1)
            print(f"  Epoch {epoch} | step {step+1}/{len(loader)} | "
                  f"loss {avg_loss:.4f} | pixel_acc {pixel_acc:.4f} | "
                  f"{time.time()-t0:.1f}s")

    avg_loss = total_loss / (len(loader) * loader.batch_size)
    pixel_acc = total_correct / max(total_pixels, 1)
    return avg_loss, pixel_acc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model. Returns (avg_loss, pixel_acc, miou)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    # Accumulate confusion for mIoU
    intersection = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        logits = F.interpolate(logits, size=masks.shape[1:],
                               mode='bilinear', align_corners=False)

        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(1)
        valid = masks != IGNORE_INDEX

        total_correct += (preds[valid] == masks[valid]).sum().item()
        total_pixels += valid.sum().item()

        # Per-class IoU accumulation
        for cls in range(NUM_CLASSES):
            pred_cls = (preds == cls) & valid
            target_cls = (masks == cls) & valid
            intersection[cls] += (pred_cls & target_cls).sum()
            union[cls] += (pred_cls | target_cls).sum()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    pixel_acc = total_correct / max(total_pixels, 1)

    # Compute mIoU (only over classes present in the dataset)
    valid_classes = union > 0
    if valid_classes.any():
        class_iou = intersection[valid_classes].float() / union[valid_classes].float()
        miou = class_iou.mean().item()
    else:
        miou = 0.0

    return avg_loss, pixel_acc, miou


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SegFormer-B0 on ADE20K")
    p.add_argument("--data-dir",
                    default=os.path.join(os.path.dirname(__file__), "data", "ADEChallengeData2016"),
                    help="Path to ADEChallengeData2016/")
    p.add_argument("--ckpt-dir",
                    default=os.path.join(os.path.dirname(__file__), "checkpoints"),
                    help="Checkpoint directory")
    p.add_argument("--epochs",       type=int,   default=160)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int,  default=5)
    p.add_argument("--crop-size",    type=int,   default=512)
    p.add_argument("--grad-accum",   type=int,   default=1,
                    help="Gradient accumulation steps")
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--print-freq",   type=int,   default=50)
    p.add_argument("--pretrained",   action="store_true",
                    help="Load pretrained HuggingFace encoder weights")
    p.add_argument("--pretrained-path", default="",
                    help="Path to pretrained checkpoint (optional)")
    p.add_argument("--resume",       default="",
                    help="Path to checkpoint to resume training from")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data:   {args.data_dir}")

    # ----- Data -----
    train_loader, val_loader = build_dataloaders(
        args.data_dir,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ----- Model -----
    model = segformer_b0(num_classes=NUM_CLASSES).to(device)

    if args.pretrained:
        ckpt_path = args.pretrained_path if args.pretrained_path else None
        load_pretrained_segformer_b0(model, checkpoint_path=ckpt_path,
                                     num_classes=NUM_CLASSES)
        print("Loaded pretrained weights")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ----- Loss / Optimizer / Scheduler -----
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Optional resume -----
    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_miou={best_miou:.4f}")

    # ----- Training loop -----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_accum=args.grad_accum, print_freq=args.print_freq,
        )

        val_loss, val_acc, val_miou = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs-1} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} mIoU {val_miou:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_miou": val_miou,
            "best_miou": best_miou,
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pth"))

        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  ✓ New best mIoU: {best_miou:.4f}")

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
