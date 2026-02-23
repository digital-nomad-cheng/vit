"""
DETR loss functions.

Includes:
- Classification loss (CE with down-weighted no-object class)
- L1 bounding box loss
- Generalized IoU (GIoU) loss
- Box conversion utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of xyxy boxes.

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        iou: (N, M) pairwise IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute Generalized IoU between two sets of xyxy boxes.

    GIoU = IoU - (area_enclosing - area_union) / area_enclosing

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        giou: (N, M) pairwise GIoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    giou = iou - (area_enc - union) / (area_enc + 1e-7)
    return giou


class DETRLoss(nn.Module):
    """DETR set-prediction loss.

    Combines classification, L1 bbox, and GIoU losses using Hungarian matching.

    Args:
        num_classes: Number of object categories (excluding no-object)
        matcher: HungarianMatcher instance
        weight_ce: Classification loss weight
        weight_bbox: L1 bbox loss weight
        weight_giou: GIoU loss weight
        eos_coef: No-object class weight (down-weighted)
    """

    def __init__(self, num_classes: int, matcher, weight_ce: float = 1.0,
                 weight_bbox: float = 5.0, weight_giou: float = 2.0,
                 eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        # Class weights: down-weight the no-object class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: dict with pred_logits (B, Q, C+1) and pred_boxes (B, Q, 4)
            targets: list of B dicts with labels (M,) and boxes (M, 4)

        Returns:
            dict of losses: loss_ce, loss_bbox, loss_giou, loss_total
        """
        # Run matcher
        indices = self.matcher(outputs, targets)

        # Classification loss
        loss_ce = self._loss_labels(outputs, targets, indices)

        # Bbox losses (only on matched pairs)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)

        loss_total = (
            self.weight_ce * loss_ce
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_total": loss_total,
        }

    def _loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Classification loss."""
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        B, Q, _ = pred_logits.shape

        # Build target labels: default = no-object (last class)
        target_classes = torch.full(
            (B, Q), self.num_classes,
            dtype=torch.int64, device=pred_logits.device,
        )
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][gt_idx]

        loss = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, C+1, Q)
            target_classes,               # (B, Q)
            weight=self.empty_weight,
        )
        return loss

    def _loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU box losses on matched pairs."""
        device = outputs["pred_boxes"].device

        # Gather matched predictions and targets
        src_boxes = []
        tgt_boxes = []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:
                src_boxes.append(outputs["pred_boxes"][b, pred_idx])
                tgt_boxes.append(targets[b]["boxes"][gt_idx].to(device))

        if not src_boxes:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero

        src_boxes = torch.cat(src_boxes)
        tgt_boxes = torch.cat(tgt_boxes)
        num_boxes = src_boxes.shape[0]

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes

        # GIoU loss
        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
        loss_giou = (1 - giou).sum() / num_boxes

        return loss_bbox, loss_giou
