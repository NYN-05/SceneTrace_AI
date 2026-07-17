"""Zero-shot object detection via Grounding DINO. Lazy-loaded, GPU-accelerated."""
import cv2
import numpy as np
import torch
import logging
from config import settings

logger = logging.getLogger("scenetrace.detector")

device = settings.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
_detector = None
_processor = None

def load():
    global _detector, _processor
    if _detector is not None:
        return
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    logger.info("Loading Grounding DINO on %s...", device)
    _detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        settings.DETECTOR_MODEL_NAME
    ).to(device).eval()
    _processor = AutoProcessor.from_pretrained(settings.DETECTOR_MODEL_NAME)
    logger.info("Grounding DINO loaded")

@torch.inference_mode()
def detect(image: np.ndarray, text: str, threshold: float = 0) -> list[dict]:
    load()
    thresh = threshold or settings.DETECTION_THRESHOLD
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = _processor(images=rgb, text=text, return_tensors="pt").to(device)
    outputs = _detector(**inputs)
    h, w = image.shape[:2]
    target = torch.tensor([[w, h]], device=device)
    results = _processor.post_process_object_detection(outputs, threshold=thresh, target_sizes=target)[0]
    dets = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        b = box.cpu().tolist()
        dets.append({
            "bbox": [int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))],
            "label": _processor.tokenizer.decode(label).strip(),
            "score": round(float(score), 3)
        })
    return dets

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
