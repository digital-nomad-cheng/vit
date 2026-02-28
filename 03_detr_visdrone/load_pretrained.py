"""Load pretrained DETR weights from Facebook Research.

Downloads the official DETR-ResNet50 checkpoint trained on COCO and loads
compatible weights into our model. The classification head (class_embed)
is skipped because COCO has 91 classes vs VisDrone's 10.

Checkpoint URL:
    https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
    - trained for 500 epochs on COCO
    - ResNet-50 backbone, d_model=256, 100 queries
    - ~160MB download
"""

import torch
from torch import nn


DETR_R50_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"


def load_pretrained_detr(model: nn.Module, num_classes: int = 10) -> None:
    """Load official Facebook DETR-R50 pretrained weights into model.

    Loads all weights except:
        - class_embed (shape mismatch: COCO 92 vs VisDrone 11)

    Args:
        model: DETR model instance (must use ResNet-50 backbone)
        num_classes: number of target classes (used for info printing)
    """
    print(f"Downloading pretrained DETR-R50 from: {DETR_R50_URL}")
    checkpoint = torch.hub.load_state_dict_from_url(
        DETR_R50_URL, map_location="cpu", check_hash=True,
    )
    pretrained_state = checkpoint["model"]

    # Get our model's state dict for comparison
    model_state = model.state_dict()

    # Separate keys into loadable and skipped
    loaded_keys = []
    skipped_keys = []

    filtered_state = {}
    for k, v in pretrained_state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(
                    f"  {k}: pretrained {list(v.shape)} vs model {list(model_state[k].shape)}"
                )
        else:
            skipped_keys.append(f"  {k}: not in model")

    # Find keys in model but not in pretrained (newly initialized)
    missing_keys = [k for k in model_state if k not in pretrained_state]

    # Load the compatible weights
    model.load_state_dict(filtered_state, strict=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Pretrained DETR-R50 weight loading summary:")
    print(f"  ✓ Loaded:           {len(loaded_keys)} parameters")
    print(f"  ✗ Skipped (shape):  {len(skipped_keys)} parameters")
    print(f"  ○ New (random init): {len(missing_keys)} parameters")

    if skipped_keys:
        print(f"\nSkipped parameters (shape mismatch or not in model):")
        for s in skipped_keys:
            print(s)

    if missing_keys:
        print(f"\nNewly initialized parameters (not in pretrained):")
        for k in missing_keys:
            print(f"  {k}: {list(model_state[k].shape)}")

    print(f"{'='*60}\n")
