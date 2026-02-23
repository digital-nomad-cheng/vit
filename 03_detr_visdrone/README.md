# DETR on VisDrone-DET

Train a [DETR](https://arxiv.org/abs/2005.12872) (DEtection TRansformer) from scratch on the [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) dataset for drone-based object detection.

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
# Default: 150 epochs, batch 4, grad_accum 4 (fits 8 GB VRAM)
uv run python 03_detr_visdrone/train.py

# Custom settings
uv run python 03_detr_visdrone/train.py --epochs 300 --batch-size 2 --lr 1e-4

# Resume from checkpoint
uv run python 03_detr_visdrone/train.py --resume ./03_detr_visdrone/checkpoints/last.pth
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
Image → ResNet-50 → 1×1 Conv (256-d) + 2D Positional Encoding
      → Transformer Encoder (6 layers)
      → Transformer Decoder (6 layers, 100 object queries)
      → Classification Head (11 classes) + BBox MLP (4 coords)
```

**Parameters**: ~41.5M

## Project Structure

```
03_detr_visdrone/
├── download_visdrone.py   # Download dataset from Google Drive
├── visdrone2coco.py       # Convert annotations to COCO format
├── dataset.py             # Dataset, transforms, collate_fn
├── detr.py                # DETR model
├── matcher.py             # Hungarian bipartite matcher
├── losses.py              # CE + L1 + GIoU loss
├── train.py               # Training loop
├── evaluate.py            # COCO mAP evaluation
└── visualize.py           # Draw predictions on images
```
