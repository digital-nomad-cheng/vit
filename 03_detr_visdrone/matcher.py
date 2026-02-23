"""
Hungarian Matcher for DETR.

Uses scipy.optimize.linear_sum_assignment to find the optimal bipartite
matching between predicted and ground-truth objects.

Cost = λ_cls * class_cost + λ_L1 * L1_cost + λ_giou * GIoU_cost
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from losses import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(torch.nn.Module):
    """Bipartite matcher between predictions and ground truth.

    Args:
        cost_class: Weight for classification cost
        cost_bbox: Weight for L1 bounding box cost
        cost_giou: Weight for GIoU cost
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: dict with:
                pred_logits: (B, num_queries, num_classes + 1)
                pred_boxes:  (B, num_queries, 4) in cxcywh [0,1]
            targets: list of B dicts, each with:
                labels: (M,) category indices
                boxes:  (M, 4) in cxcywh [0,1]

        Returns:
            list of (pred_indices, gt_indices) tuples, one per batch element
        """
        B, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten across batch for efficiency
        pred_logits = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (B*Q, C+1)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)               # (B*Q, 4)

        # Concatenate targets
        tgt_labels = torch.cat([t["labels"] for t in targets])         # (sum(M),)
        tgt_boxes = torch.cat([t["boxes"] for t in targets])           # (sum(M), 4)

        if tgt_labels.numel() == 0:
            # No ground truth in this batch
            return [(torch.tensor([], dtype=torch.int64),
                     torch.tensor([], dtype=torch.int64)) for _ in range(B)]

        # Classification cost: -prob of correct class
        cost_class = -pred_logits[:, tgt_labels]  # (B*Q, sum(M))

        # L1 bbox cost
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)  # (B*Q, sum(M))

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )  # (B*Q, sum(M))

        # Total cost
        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou)

        # Reshape to per-image and run Hungarian algorithm
        C = C.view(B, num_queries, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, dim=-1)):
            ci = c[i]  # (num_queries, M_i)
            pred_idx, gt_idx = linear_sum_assignment(ci.numpy())
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(gt_idx, dtype=torch.int64),
            ))

        return indices
