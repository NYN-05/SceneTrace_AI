"""Zero-shot object detection — YOLO-World (default) with automatic fallback,
or Grounding DINO.
Set DETECTOR_BACKEND=grounding_dino in .env to use Grounding DINO instead.
YOLO-World fallback chain: Large -> Medium -> Small.
"""
import threading
from collections import OrderedDict
from typing import ClassVar

import cv2
import numpy as np
import torch
from config import logger, settings

device = settings.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

# ---- Grounding DINO globals ----
_detector = None
_processor = None

# ---- YOLO-World manager ----
_yolo_manager_lock = threading.Lock()
_yolo_bool_lock = threading.Lock()


_YOLO_FILE_MAP = {
    "large": "yolov8l-world.pt",
    "medium": "yolov8m-world.pt",
    "small": "yolov8s-world.pt",
}


class YOLOWorldManager:
    VARIANTS: ClassVar[OrderedDict] = OrderedDict()
    for _v in settings.YOLO_WORLD_FALLBACKS.split(","):
        _v = _v.strip()
        if _v in _YOLO_FILE_MAP:
            VARIANTS[_v] = _YOLO_FILE_MAP[_v]

    def __init__(self, default_variant="large"):
        self.default = default_variant if default_variant in self.VARIANTS else next(iter(self.VARIANTS))
        self._models = {}
        self._current = None
        self._load_lock = threading.Lock()

    @property
    def current_variant(self):
        return self._current

    def _load(self, name: str):
        from ultralytics import YOLOWorld
        variant_label = name.capitalize()
        logger.info("Loading YOLO-World-%s (%s)...", variant_label, self.VARIANTS[name])
        model = YOLOWorld(self.VARIANTS[name])
        if _has_meta_tensor(model):
            _reload_model_weights(model, self.VARIANTS[name], device)
        _move_to_device(model, device)
        _purge_meta_tensors(model, device)
        actual = next(model.parameters(), None)
        logger.info("YOLO-World-%s loaded on %s", variant_label, actual.device if actual else "?")
        return model

    def get(self, name: str = None):
        name = name or self.default
        with self._load_lock:
            if name not in self._models:
                self._models[name] = self._load(name)
        return self._models[name]

    def detect(self, image: np.ndarray, text: str, threshold: float) -> list[dict]:
        names = list(self.VARIANTS.keys())
        start = names.index(self.default)
        ordered = names[start:] + names[:start]
        seen = set()
        chain = [v for v in ordered if not (v in seen or seen.add(v))]

        last_error = None
        for name in chain:
            try:
                model = self.get(name)
                model.model.conf = float(threshold or settings.DETECTION_THRESHOLD)

                texts = [t.strip() for t in text.split(",") if t.strip()]
                model.set_classes(texts)

                with _yolo_bool_lock:
                    orig_bool = torch.Tensor.__bool__
                    def _safe_bool(self):
                        if self.numel() != 1:
                            return self.numel() > 0
                        return orig_bool(self)
                    torch.Tensor.__bool__ = _safe_bool
                    try:
                        results = model(image, verbose=False)
                    finally:
                        torch.Tensor.__bool__ = orig_bool
                r = results[0]

                h, w = image.shape[:2]
                dets = []
                if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes) > 0:
                    for i in range(len(r.boxes)):
                        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().tolist()
                        ci = int(r.boxes.cls[i].item())
                        dets.append({
                            "bbox": [int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))],
                            "label": texts[ci] if ci < len(texts) else text,
                            "score": round(float(r.boxes.conf[i].item()), 3),
                        })

                if name != self.default:
                    logger.info("Fell back to YOLO-World-%s (was using %s)", name.capitalize(), self.default.capitalize())
                self._current = name
                return dets

            except Exception as exc:
                last_error = exc
                logger.warning("YOLO-World-%s failed: %s", name.capitalize(), exc)

        msg = f"All YOLO-World variants failed — {last_error}"
        raise RuntimeError(msg)


def _reload_model_weights(model, variant_name: str, target_device: str):
    import ultralytics.utils.downloads
    ckpt_path = ultralytics.utils.downloads.attempt_download_asset(variant_name)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.to_empty(device=target_device)
    raw = ckpt.get("model", ckpt)
    if isinstance(raw, dict):
        model.model.load_state_dict(raw, strict=False)
    elif hasattr(raw, "state_dict"):
        model.model.load_state_dict(raw.state_dict(), strict=False)


def _move_to_device(module: torch.nn.Module, target: str):
    if _has_meta_tensor(module):
        return
    for p in module.parameters(recurse=True):
        if p.device.type != target.split(":")[0]:
            p.data = p.data.to(target)
    for b in module.buffers(recurse=True):
        if b.device.type != target.split(":")[0]:
            b.data = b.data.to(target)


def _has_meta_tensor(module: torch.nn.Module) -> bool:
    if any(p.device.type == "meta" for p in module.parameters()):
        return True
    return any(b.device.type == "meta" for b in module.buffers())


def _purge_meta_tensors(module: torch.nn.Module, target: str):
    for p in module.parameters(recurse=True):
        if p.device.type == "meta":
            p.data = torch.empty(p.size(), dtype=p.dtype, device=target)
    for b in module.buffers(recurse=True):
        if b.device.type == "meta":
            b.data = torch.empty(b.size(), dtype=b.dtype, device=target)


_yolo_manager = None


def _get_yolo():
    global _yolo_manager
    if _yolo_manager is None:
        with _yolo_manager_lock:
            if _yolo_manager is None:
                _yolo_manager = YOLOWorldManager(settings.YOLO_WORLD_DEFAULT)
    return _yolo_manager


# =============================================================================
#  Grounding DINO backend
# =============================================================================

def _load_grounding_dino():
    global _detector, _processor
    if _detector is not None:
        return
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    logger.info("Loading Grounding DINO on %s...", device)
    _detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        settings.DETECTOR_MODEL_NAME,
    ).eval().to(device)
    _processor = AutoProcessor.from_pretrained(settings.DETECTOR_MODEL_NAME)
    logger.info("Grounding DINO loaded")


def _detect_grounding_dino(image: np.ndarray, text: str, threshold: float) -> list[dict]:
    global _detector, _processor
    thresh = threshold or settings.DETECTION_THRESHOLD
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    texts = [t.strip() for t in text.split(",") if t.strip()]
    inputs = _processor(images=rgb, text=text, return_tensors="pt").to(device)
    outputs = _detector(**inputs)
    h, w = image.shape[:2]
    target = torch.tensor([[w, h]], device=device)
    results = _processor.post_process_grounded_object_detection(
        outputs, threshold=thresh, target_sizes=target, text_labels=[texts],
    )[0]
    dets = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        b = box.cpu().tolist()
        if isinstance(label, str):
            lbl = label
        else:
            ci = int(label.item()) if torch.is_tensor(label) else int(label)
            lbl = texts[ci] if ci < len(texts) else text
        dets.append({
            "bbox": [int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))],
            "label": lbl,
            "score": round(float(score), 3),
        })
    return dets


# =============================================================================
#  Public API (identical interface regardless of backend)
# =============================================================================

def load():
    if settings.DETECTOR_BACKEND == "yolo_world":
        _get_yolo()
    else:
        _load_grounding_dino()


@torch.inference_mode()
def detect(image: np.ndarray, text: str, threshold: float = 0) -> list[dict]:
    if settings.DETECTOR_BACKEND == "yolo_world":
        return _get_yolo().detect(image, text, threshold)
    load()
    return _detect_grounding_dino(image, text, threshold)


def render(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    img = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['label']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def detect_and_save(image_path: str, text: str, output_path: str = None) -> list[dict]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    dets = detect(img, text)
    if output_path and dets:
        annotated = render(img, dets)
        cv2.imwrite(str(output_path), annotated)
    return dets
