"""
EasyPortrait semantic segmentation dataset for PyTorch.

EasyPortrait is a face parsing & portrait segmentation dataset with 9 classes.
Annotations are stored as indexed PNGs where pixel values 0–8 directly
represent class IDs (no remapping needed).

Classes:
    0: Background     1: Person       2: Face skin
    3: Left brow      4: Right brow   5: Left eye
    6: Right eye      7: Lips         8: Teeth

Expected directory layout (EasyPortrait/):
    images/
        train/    — JPEGs
        val/      — JPEGs
        test/     — JPEGs
    annotations/
        train/    — PNGs (pixel values 0–8)
        val/      — PNGs
        test/     — PNGs

Usage:
    train_loader, val_loader = build_easyportrait_dataloaders("data/EasyPortrait")

References:
    https://github.com/hukenovs/easyportrait
    https://arxiv.org/abs/2304.13509
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 9
IGNORE_INDEX = 255  # pixels with this label are ignored in loss

EASYPORTRAIT_CLASSES = [
    "background",
    "person",
    "face_skin",
    "left_brow",
    "right_brow",
    "left_eye",
    "right_eye",
    "lips",
    "teeth",
]


class EasyPortraitDataset(Dataset):
    """EasyPortrait semantic segmentation dataset.

    Masks use pixel values 0–8 directly as class IDs.
    """

    def __init__(self, root: str, split: str = "train",
                 transform=None):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)

        assert os.path.isdir(self.images_dir), \
            f"Images directory not found: {self.images_dir}"
        assert os.path.isdir(self.masks_dir), \
            f"Annotations directory not found: {self.masks_dir}"

        self.filenames = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        assert len(self.filenames) > 0, \
            f"No images found in {self.images_dir}"
        print(f"EasyPortrait {split}: {len(self.filenames)} images")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]

        # Load image
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert("RGB")

        # Load mask — annotation files use .png extension
        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_fname)
        mask = Image.open(mask_path)  # indexed PNG, values 0–8

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = TF.to_tensor(img)
            img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask


class EasyPortraitTransform:
    """Transforms for EasyPortrait segmentation (applied jointly to image + mask).

    Train: RandomResize → RandomCrop → RandomHorizontalFlip → ColorJitter → Normalize
    Val:   Resize to crop_size → Normalize
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
        if self.train:
            # Random scale
            scale = random.uniform(*self.scale_range)
            w, h = img.size
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = TF.resize(img, [new_h, new_w],
                            interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [new_h, new_w],
                             interpolation=T.InterpolationMode.NEAREST)

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
                mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=IGNORE_INDEX)

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

        # Mask: values are already 0-indexed class IDs (0–8), no remapping
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask


def build_easyportrait_dataloaders(
    data_dir: str,
    crop_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders for EasyPortrait."""
    train_ds = EasyPortraitDataset(
        data_dir, split="train",
        transform=EasyPortraitTransform(train=True, crop_size=crop_size),
    )
    val_ds = EasyPortraitDataset(
        data_dir, split="val",
        transform=EasyPortraitTransform(train=False, crop_size=crop_size),
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
