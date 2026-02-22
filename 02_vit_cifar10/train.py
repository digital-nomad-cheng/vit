"""
Train a vision transformer model on CIFAR-10.

CIFAR-10 images are 32x32 RGB — a natural fit for patch_size=4
(8x8 = 64 patches, no resizing required).

Usage:
    uv run python 02_vit_cifar10/train.py --model vit
    uv run python 02_vit_cifar10/train.py --epochs 100 --batch-size 128 --lr 5e-4
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(__file__))
from tnt import TNT
from vit import ViT


def build_tnt(num_classes: int = 10) -> TNT:
    return TNT(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        outer_dim=192,        # sentence (patch) embedding dim
        inner_dim=16,         # word (pixel-group) embedding dim
        depth=6,              # number of TNT blocks
        outer_num_heads=3,    # must divide outer_dim
        inner_num_heads=4,    # must divide inner_dim
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        inner_stride=4,
    )

def build_vit(num_classes: int = 10) -> ViT:
    return ViT(
        image_size=32,
        patch_size=4,
        num_classes=num_classes,
        dim=192,
        depth=6,
        heads=3,
        mlp_dim=768,
        dropout=0.1,
        pool='cls',
    )


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, print_freq: int = 100):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        if (step + 1) % print_freq == 0:
            print(f"  Epoch {epoch} | step {step+1}/{len(loader)} | "
                  f"loss {total_loss/total:.4f} | acc {correct/total:.4f} | "
                  f"{time.time()-t0:.1f}s")

    return total_loss / total, correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Train TNT on CIFAR-10")
    p.add_argument("--data-dir",     default="./.data",               help="Dataset directory")
    p.add_argument("--ckpt-dir",     default="./.data/checkpoints/", help="Checkpoint directory")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs",type=int,   default=5)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--resume",       default="", help="Path to checkpoint to resume from")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model",        choices=["tnt", "vit"], default="tnt", help="Model architecture to use")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device : {device}")

    # ----- Data -----
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ----- Model -----
    if args.model == "tnt":
        model = build_tnt(num_classes=10).to(device)
        args.checkpoint_dir = os.path.join(args.ckpt_dir, "tnt_cifar10")
    elif args.model == "vit":
        model = build_vit(num_classes=10).to(device)
        args.checkpoint_dir = os.path.join(args.ckpt_dir, "vit_cifar10")
    else: 
        raise ValueError(f"Unknown model: {args.model}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ----- Loss / Optimiser / Scheduler -----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Optional resume -----
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # ----- Training loop -----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs-1} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {current_lr:.2e} | {time.time()-t0:.1f}s")

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_acc": val_acc,
            "best_acc": best_acc,
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pth"))

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  ✓ New best val acc: {best_acc:.4f}")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    print(f"Checkpoints saved to: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
