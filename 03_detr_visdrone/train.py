"""
Train DETR on VisDrone-DET dataset.

Usage:
    uv run python 03_detr_visdrone/train.py
    uv run python 03_detr_visdrone/train.py --pretrained-detr --epochs 50
    uv run python 03_detr_visdrone/train.py --resume ./03_detr_visdrone/checkpoints/last.pth
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_dataloaders
from detr import build_detr
from matcher import HungarianMatcher
from losses import DETRLoss
from load_pretrained import load_pretrained_detr
from config import NUM_CLASSES



def train_one_epoch(
    model: nn.Module,
    criterion: DETRLoss,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum: int = 1,
    print_freq: int = 50,
    max_grad_norm: float = 0.1,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    criterion.train()

    total_loss = 0.0
    total_ce = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    num_batches = 0
    t0 = time.time()

    optimizer.zero_grad()

    for step, (images, masks, targets) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward
        outputs = model(images, masks)
        losses = criterion(outputs, targets)
        loss = losses["loss_total"] / grad_accum

        # Backward
        loss.backward()

        if (step + 1) % grad_accum == 0:
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += losses["loss_total"].item()
        total_ce += losses["loss_ce"].item()
        total_bbox += losses["loss_bbox"].item()
        total_giou += losses["loss_giou"].item()
        num_batches += 1

        if (step + 1) % print_freq == 0:
            avg_loss = total_loss / num_batches
            print(
                f"  Epoch {epoch} | step {step+1}/{len(loader)} | "
                f"loss {avg_loss:.4f} (ce={total_ce/num_batches:.4f} "
                f"bbox={total_bbox/num_batches:.4f} "
                f"giou={total_giou/num_batches:.4f}) | "
                f"{time.time()-t0:.1f}s"
            )

    # Flush remaining gradients
    if len(loader) % grad_accum != 0:
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_ce": total_ce / max(num_batches, 1),
        "loss_bbox": total_bbox / max(num_batches, 1),
        "loss_giou": total_giou / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    criterion: DETRLoss,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Compute validation loss."""
    model.eval()
    criterion.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    num_batches = 0

    for images, masks, targets in loader:
        images = images.to(device)
        masks = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images, masks)
        losses = criterion(outputs, targets)

        total_loss += losses["loss_total"].item()
        total_ce += losses["loss_ce"].item()
        total_bbox += losses["loss_bbox"].item()
        total_giou += losses["loss_giou"].item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_ce": total_ce / max(num_batches, 1),
        "loss_bbox": total_bbox / max(num_batches, 1),
        "loss_giou": total_giou / max(num_batches, 1),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train DETR on VisDrone-DET")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"),
                    help="Path to VisDrone data directory")
    p.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"),
                    help="Checkpoint save directory")
    p.add_argument("--epochs", type=int, default=150, help="Total training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (transformer)")
    p.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate (backbone)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--lr-drop", type=int, default=100, help="Epoch to drop LR by 0.1x")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--max-grad-norm", type=float, default=0.1, help="Max gradient norm for clipping")
    p.add_argument("--num-queries", type=int, default=100, help="Number of object queries")
    p.add_argument("--max-size", type=int, default=800, help="Max image size")
    p.add_argument("--pretrained-detr", action="store_true", default=False,
                    help="Load pretrained DETR-R50 weights (COCO-trained)")
    p.add_argument("--freeze-all", action="store_true", default=False,
                    help="Freeze all layers except class_embed (head-only fine-tuning)")
    p.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    p.add_argument("--eval-freq", type=int, default=5, help="Evaluate every N epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, lr_backbone={args.lr_backbone}, "
          f"grad_accum={args.grad_accum}, num_queries={args.num_queries}")

    # ----- Data -----
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_size=args.max_size,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ----- Model -----
    model = build_detr(
        num_classes=NUM_CLASSES,
        pretrained_backbone=not args.pretrained_detr,  # skip if loading full DETR
        num_queries=args.num_queries,
    ).to(device)

    # Load pretrained DETR weights (COCO-trained)
    if args.pretrained_detr:
        load_pretrained_detr(model, num_classes=NUM_CLASSES)

    # Freeze all layers except class_embed (head-only fine-tuning)
    if args.freeze_all:
        for name, param in model.named_parameters():
            if "class_embed" not in name:
                param.requires_grad_(False)
        print("Frozen all layers except class_embed")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ----- Loss -----
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = DETRLoss(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_ce=1.0, weight_bbox=5.0, weight_giou=2.0,
        eos_coef=0.1,
    ).to(device)

    # ----- Optimizer -----
    # Different LR for backbone vs transformer/heads
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    # ----- Resume -----
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ----- Training loop -----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch=epoch, grad_accum=args.grad_accum,
        )

        scheduler.step()

        # Print training metrics
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_tr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs-1} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"(ce={train_metrics['loss_ce']:.4f} "
            f"bbox={train_metrics['loss_bbox']:.4f} "
            f"giou={train_metrics['loss_giou']:.4f}) | "
            f"lr_bb={lr_bb:.2e} lr_tr={lr_tr:.2e} | "
            f"{time.time()-t0:.1f}s"
        )

        # Evaluate
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate_loss(model, criterion, val_loader, device)
            print(
                f"  Val loss {val_metrics['loss']:.4f} "
                f"(ce={val_metrics['loss_ce']:.4f} "
                f"bbox={val_metrics['loss_bbox']:.4f} "
                f"giou={val_metrics['loss_giou']:.4f})"
            )
            val_loss = val_metrics["loss"]
        else:
            val_loss = None

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "best_loss": best_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pth"))

        # Save best
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            ckpt["best_loss"] = best_loss
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  ✓ New best val loss: {best_loss:.4f}")

    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
