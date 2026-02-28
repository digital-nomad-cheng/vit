# DETR on VisDrone-DET

Fine-tune a [DETR](https://arxiv.org/abs/2005.12872) (DEtection TRansformer) on the [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) dataset for drone-based object detection, using Facebook's COCO-pretrained weights.

The architecture matches the [official DETR implementation](https://github.com/facebookresearch/detr) exactly, enabling direct pretrained weight loading.

## Classes

| ID | Category | ID | Category |
|----|----------|-----|----------|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

## Setup

```bash
# 1. Download VisDrone-DET trainset (~1.44 GB) and valset (~0.07 GB)
uv run python 03_detr_visdrone/download_visdrone.py

# 2. Convert VisDrone annotations to COCO JSON format
uv run python 03_detr_visdrone/visdrone2coco.py
```

## Training

```bash
# Recommended: Fine-tune from COCO-pretrained DETR-R50 (~160MB download, cached)
uv run python 03_detr_visdrone/train.py --pretrained-detr --epochs 50

# Head-only fine-tune (fastest — only trains classification head, 2,827 params)
uv run python 03_detr_visdrone/train.py --pretrained-detr --freeze-all --lr 1e-3 --epochs 10

# Then full fine-tune from the head-only checkpoint
uv run python 03_detr_visdrone/train.py --pretrained-detr \
    --resume ./03_detr_visdrone/checkpoints/last.pth \
    --lr 1e-5 --lr-backbone 1e-6 --epochs 50

# Train from scratch (not recommended — very slow convergence)
uv run python 03_detr_visdrone/train.py --epochs 300
```

## Evaluation

```bash
# Compute COCO mAP on validation set
uv run python 03_detr_visdrone/evaluate.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth
```

## Visualization

```bash
# Draw predictions on random val images
uv run python 03_detr_visdrone/visualize.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth

# Specific images with custom threshold
uv run python 03_detr_visdrone/visualize.py --checkpoint ./03_detr_visdrone/checkpoints/best.pth \
    --images img1.jpg img2.jpg --score-thresh 0.5
```

## Architecture

```
Image → ResNet-50 (FrozenBatchNorm2d) → 1×1 Conv (256-d)
      → PositionEmbeddingSine (2D sinusoidal, normalized)
      → Transformer Encoder (6 layers, post-norm)
      → Transformer Decoder (6 layers, 100 object queries)
      → class_embed (11 classes) + bbox_embed MLP (4 coords)
```

**Parameters**: ~41.5M (pretrained from COCO, class_embed randomly initialized)

## Project Structure

```
03_detr_visdrone/
├── config.py              # Shared constants (classes, normalization)
├── download_visdrone.py   # Download dataset from Google Drive
├── visdrone2coco.py       # Convert annotations to COCO format
├── dataset.py             # Dataset, transforms, collate_fn
├── detr.py                # DETR model (official architecture)
├── load_pretrained.py     # Download & load COCO-pretrained weights
├── matcher.py             # Hungarian bipartite matcher
├── losses.py              # CE + L1 + GIoU loss
├── train.py               # Training loop
├── evaluate.py            # COCO mAP evaluation
└── visualize.py           # Draw predictions on images
```
