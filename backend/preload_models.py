"""Pre-download and cache all ML models so they aren't downloaded during testing or first run.

Usage:
    pip install hf_transfer  # optional: Rust-based downloader (10-50x faster)
    python preload_models.py
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass

from config import logger, settings


def _download_model(model_id: str, model_type: str):
    logger.info("Downloading %s model: %s ...", model_type, model_id)
    if model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        CLIPModel.from_pretrained(model_id)
        CLIPProcessor.from_pretrained(model_id)
    elif model_type == "dino":
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, resume_download=True, ignore_mismatched_sizes=False,
        )
        AutoProcessor.from_pretrained(model_id, resume_download=True)
    logger.info("%s model cached (%s)", model_type, model_id)


def _preload_yolo_world(variant: str):
    logger.info("Downloading YOLO-World-%s ...", variant.capitalize())
    from ultralytics import YOLOWorld
    model_ids = {"large": "yolov8l-world.pt", "medium": "yolov8m-world.pt", "small": "yolov8s-world.pt"}
    YOLOWorld(model_ids[variant])
    logger.info("YOLO-World-%s cached", variant.capitalize())


def preload_all(variant: str = None):
    logger.info("=== Preloading all ML models ===")
    _download_model(settings.CLIP_MODEL_NAME, "clip")
    if settings.DETECTOR_BACKEND == "yolo_world":
        v = variant or settings.YOLO_WORLD_DEFAULT
        _preload_yolo_world(v)
    else:
        _download_model(settings.DETECTOR_MODEL_NAME, "dino")
    logger.info("=== All models preloaded and cached ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download all ML models")
    parser.add_argument("--variant", default=None, choices=["small", "medium", "large"],
                        help="YOLO-World variant to preload (default: from .env)")
    args = parser.parse_args()
    preload_all(variant=args.variant)
