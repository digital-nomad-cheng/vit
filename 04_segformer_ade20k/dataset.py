"""
ADE20K semantic segmentation dataset for PyTorch.

ADE20K contains 20,210 training and 2,000 validation images with
150 semantic categories. Annotations are stored as indexed PNGs where
pixel value 0 = unlabeled/background and 1-150 = class IDs.

Expected directory layout (ADEChallengeData2016/):
    images/
        training/    — 20,210 JPEGs
        validation/  — 2,000 JPEGs
    annotations/
        training/    — 20,210 PNGs
        validation/  — 2,000 PNGs

Usage:
    train_loader, val_loader = build_dataloaders("data/ADEChallengeData2016")
"""

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 150
IGNORE_INDEX = 255  # pixels with this label are ignored in loss

# ADE20K class names (0-indexed, 150 classes)
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person",
    "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers",
    "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway",
    "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair",
    "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "van",
    "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave",
    "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
]


class ADE20KDataset(Dataset):
    """ADE20K semantic segmentation dataset.

    Masks use:
        0       → unlabeled (mapped to IGNORE_INDEX=255)
        1–150   → class IDs (mapped to 0–149 for 0-indexed)
    """

    def __init__(self, root: str, split: str = "training",
                 transform=None):
        assert split in ("training", "validation")
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)

        self.filenames = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

        assert len(self.filenames) > 0, \
            f"No images found in {self.images_dir}"
        print(f"ADE20K {split}: {len(self.filenames)} images")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]

        # Load image
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert("RGB")

        # Load mask — annotation files use .png extension
        mask_fname = fname.replace(".jpg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_fname)
        mask = Image.open(mask_path)  # indexed PNG, values 0-150

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = TF.to_tensor(img)
            img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
            mask = torch.as_tensor(
                __import__("numpy").array(mask), dtype=torch.long
            )

        return img, mask


class SegTransform:
    """Transforms for semantic segmentation (applied jointly to image + mask).

    Train: RandomResize → RandomCrop → RandomHorizontalFlip → ColorJitter → Normalize
    Val: Resize to crop_size → Normalize
    """

    def __init__(self, train: bool = True, crop_size: int = 512,
                 scale_range: tuple[float, float] = (0.5, 2.0)):
        self.train = train
        self.crop_size = crop_size
        self.scale_range = scale_range

        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if train:
            self.color_jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )

    def __call__(self, img: Image.Image, mask: Image.Image
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        if self.train:
            # Random scale
            scale = random.uniform(*self.scale_range)
            w, h = img.size
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = TF.resize(img, [new_h, new_w], interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [new_h, new_w], interpolation=T.InterpolationMode.NEAREST)

            # Random crop
            w, h = img.size
            crop_h = min(h, self.crop_size)
            crop_w = min(w, self.crop_size)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            img = TF.crop(img, top, left, crop_h, crop_w)
            mask = TF.crop(mask, top, left, crop_h, crop_w)

            # Pad if smaller than crop_size
            w, h = img.size
            if h < self.crop_size or w < self.crop_size:
                pad_h = max(self.crop_size - h, 0)
                pad_w = max(self.crop_size - w, 0)
                img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
                mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)

            # Random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Color jitter (image only)
            img = self.color_jitter(img)
        else:
            # Resize to crop_size for validation
            img = TF.resize(img, [self.crop_size, self.crop_size],
                            interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.crop_size, self.crop_size],
                             interpolation=T.InterpolationMode.NEAREST)

        # To tensor + normalize
        img = TF.to_tensor(img)
        img = self.normalize(img)

        # Mask: convert to long tensor, remap 0 → ignore, 1-150 → 0-149
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask[mask == 0] = IGNORE_INDEX  # unlabeled → ignore
        mask[mask != IGNORE_INDEX] -= 1  # 1-150 → 0-149

        return img, mask


def build_dataloaders(
    data_dir: str,
    crop_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders for ADE20K."""
    train_ds = ADE20KDataset(
        data_dir, split="training",
        transform=SegTransform(train=True, crop_size=crop_size),
    )
    val_ds = ADE20KDataset(
        data_dir, split="validation",
        transform=SegTransform(train=False, crop_size=crop_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
