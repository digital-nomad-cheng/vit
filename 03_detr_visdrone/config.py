"""Shared constants for the VisDrone-DET DETR project."""

NUM_CLASSES = 10

CATEGORY_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# COCO-format category dicts (used by visdrone2coco.py)
VISDRONE_CATEGORIES = [
    {"id": i, "name": name} for i, name in enumerate(CATEGORY_NAMES)
]

# ImageNet normalization (used by dataset.py and visualize.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
