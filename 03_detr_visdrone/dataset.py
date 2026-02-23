"""
COCO-format PyTorch Dataset for VisDrone-DET with DETR-style transforms.

Provides:
- VisDroneCOCO dataset class
- Training and validation transforms (resize, flip, normalize)
- Custom collate_fn that pads images and creates pixel masks
"""

import os
import random
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF


from config import IMAGENET_MEAN, IMAGENET_STD


class VisDroneCOCO(Dataset):
    """COCO-format dataset for VisDrone object detection.

    Each sample returns (image_tensor, target_dict) where target_dict contains:
        - boxes: (N, 4) in cxcywh format, normalized to [0,1]
        - labels: (N,) category indices
        - image_id: scalar tensor
        - orig_size: (2,) tensor [h, w]
    """

    def __init__(self, images_dir: str, ann_file: str, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Convert boxes from xywh to cxcywh normalized
        boxes = []
        labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            # Convert to center format and normalize
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            boxes.append([cx, cy, nw, nh])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.tensor([h, w]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class DetrTransform:
    """DETR-style image and target transforms."""

    def __init__(self, train: bool = True, max_size: int = 800,
                 scales: list[int] | None = None):
        self.train = train
        self.max_size = max_size
        self.scales = scales or list(range(480, 801, 32))  # 480, 512, ..., 800

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        if self.train:
            # Random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                boxes = target["boxes"]
                # Flip cx: cx_new = 1 - cx
                boxes[:, 0] = 1.0 - boxes[:, 0]
                target["boxes"] = boxes

            # Random resize
            size = random.choice(self.scales)
        else:
            size = self.max_size

        # Resize keeping aspect ratio
        w, h = img.size
        scale = size / max(h, w)
        if scale < 1.0 or self.train:
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = TF.resize(img, [new_h, new_w])

        # Normalize
        img = self.normalize(img)
        return img, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Pad images to the same size and create pixel masks.

    Returns:
        images: (B, 3, max_H, max_W) padded tensor
        masks:  (B, max_H, max_W) binary mask (True = padding)
        targets: list of target dicts
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    batch_images = []
    batch_masks = []

    for img in images:
        _, h, w = img.shape
        # Pad to max size (right and bottom)
        pad_h = max_h - h
        pad_w = max_w - w
        padded = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        batch_images.append(padded)

        # Mask: True where padded
        mask = torch.ones(max_h, max_w, dtype=torch.bool)
        mask[:h, :w] = False
        batch_masks.append(mask)

    images_tensor = torch.stack(batch_images)
    masks_tensor = torch.stack(batch_masks)

    return images_tensor, masks_tensor, targets


def build_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_size: int = 800,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders."""

    train_images = os.path.join(data_dir, "VisDrone2019-DET-train", "images")
    val_images = os.path.join(data_dir, "VisDrone2019-DET-val", "images")
    train_ann = os.path.join(data_dir, "train.json")
    val_ann = os.path.join(data_dir, "val.json")

    train_ds = VisDroneCOCO(
        train_images, train_ann,
        transforms=DetrTransform(train=True, max_size=max_size),
    )
    val_ds = VisDroneCOCO(
        val_images, val_ann,
        transforms=DetrTransform(train=False, max_size=max_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
