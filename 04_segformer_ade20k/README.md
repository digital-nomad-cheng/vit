# SegFormer-B0 on ADE20K

From-scratch implementation of [SegFormer](https://arxiv.org/abs/2105.15203) (NeurIPS 2021) for semantic segmentation on the [ADE20K](http://sceneparsing.csail.mit.edu/) dataset (150 classes, 20K train / 2K val images).

The model uses a **Mix Transformer (MiT-B0)** hierarchical encoder with an **All-MLP decoder** — no positional encodings, no complex decoders.

## Architecture

```
Image (3×H×W)
  → Stage 1: OverlapPatchEmbed(7×7, s=4) → 2× TransformerBlock → 32×H/4×W/4
  → Stage 2: OverlapPatchEmbed(3×3, s=2) → 2× TransformerBlock → 64×H/8×W/8
  → Stage 3: OverlapPatchEmbed(3×3, s=2) → 2× TransformerBlock → 160×H/16×W/16
  → Stage 4: OverlapPatchEmbed(3×3, s=2) → 2× TransformerBlock → 256×H/32×W/32
  → MLP Decoder: project all stages → 256-d, upsample to H/4, concat, fuse
  → Output: 150×H/4×W/4 (upsample to full res for final prediction)
```

Each TransformerBlock: `LayerNorm → EfficientSelfAttention → LayerNorm → MixFFN`

- **EfficientSelfAttention**: reduces K,V sequence length by ratio R (8,4,2,1 per stage)
- **MixFFN**: 3×3 depth-wise conv replaces positional encoding

**Parameters**: 3.75M (3.3M encoder + 0.4M decoder)

## Setup

```bash
# Download ADE20K dataset (~923 MB)
uv run python 04_segformer_ade20k/download_ade20k.py
```

## Training

```bash
# Train from scratch
uv run python 04_segformer_ade20k/train.py --epochs 160 --batch-size 8

# Fine-tune from HuggingFace pretrained weights (needs: uv pip install huggingface_hub)
uv run python 04_segformer_ade20k/train.py --pretrained --epochs 50 --lr 6e-5

# Resume training from checkpoint
uv run python 04_segformer_ade20k/train.py --resume ./04_segformer_ade20k/checkpoints/last.pth

# Smaller GPU / gradient accumulation
uv run python 04_segformer_ade20k/train.py --batch-size 2 --grad-accum 4
```

## MiT-B0 Encoder Details

| Stage | Patch Size | Stride | Channels | Layers | SR Ratio | Heads | FFN Ratio |
|-------|-----------|--------|----------|--------|----------|-------|-----------|
| 1     | 7×7       | 4      | 32       | 2      | 8        | 1     | 4         |
| 2     | 3×3       | 2      | 64       | 2      | 4        | 2     | 4         |
| 3     | 3×3       | 2      | 160      | 2      | 2        | 5     | 4         |
| 4     | 3×3       | 2      | 256      | 2      | 1        | 8     | 4         |

## Project Structure

```
04_segformer_ade20k/
├── segformer.py          # SegFormer-B0 model (encoder + decoder)
├── dataset.py            # ADE20K dataset, transforms, dataloaders
├── train.py              # Training loop with mIoU evaluation
├── download_ade20k.py    # Download ADE20K dataset
└── README.md
```
