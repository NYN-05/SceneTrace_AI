import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "20/minute")
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", str(Path(__file__).parent / "storage")))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEVICE: str = os.getenv("DEVICE", "")

    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    DETECTOR_MODEL_NAME: str = os.getenv("DETECTOR_MODEL_NAME", "IDEA-Research/grounding-dino-base")

    DETECTOR_BACKEND: str = os.getenv("DETECTOR_BACKEND", "yolo_world")
    YOLO_WORLD_DEFAULT: str = os.getenv("YOLO_WORLD_DEFAULT", "large")
    YOLO_WORLD_FALLBACKS: str = os.getenv("YOLO_WORLD_FALLBACKS", "large,medium,small")

    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.55"))
    OBJECT_WEIGHT: float = float(os.getenv("OBJECT_WEIGHT", "0.45"))
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.2"))
    SEARCH_HIGH_THRESHOLD: float = float(os.getenv("SEARCH_HIGH_THRESHOLD", "0.25"))
    SEARCH_MEDIUM_THRESHOLD: float = float(os.getenv("SEARCH_MEDIUM_THRESHOLD", "0.15"))
    GAP_THRESHOLD: int = int(os.getenv("GAP_THRESHOLD", "3"))
    MOTION_STRIDE: int = int(os.getenv("MOTION_STRIDE", "3"))
    MOTION_TARGET_PCT: float = float(os.getenv("MOTION_TARGET_PCT", "5.0"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
    THUMBNAIL_QUALITY: int = int(os.getenv("THUMBNAIL_QUALITY", "80"))
    DET_CACHE_MAXSIZE: int = int(os.getenv("DET_CACHE_MAXSIZE", "5000"))
    DET_CACHE_TTL: int = int(os.getenv("DET_CACHE_TTL", "3600"))
    FAISS_IVF_THRESHOLD: int = int(os.getenv("FAISS_IVF_THRESHOLD", "50"))

    INDEX_DETECTION_PROMPTS: str = os.getenv("INDEX_DETECTION_PROMPTS",
        "person, car, bicycle, motorcycle, bus, truck, traffic light, fire hydrant, "
        "stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
        "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, "
        "baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, "
        "fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, "
        "donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, "
        "keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, "
        "teddy bear, hair drier, toothbrush")
    INDEX_MIN_FPS: int = int(os.getenv("INDEX_MIN_FPS", "5"))
    INDEX_OBJECT_CONFIDENCE: float = float(os.getenv("INDEX_OBJECT_CONFIDENCE", "0.3"))
    INDEX_OBJECT_EMBED_DIM: int = int(os.getenv("INDEX_OBJECT_EMBED_DIM", "512"))

    API_KEY: str = os.getenv("API_KEY", "")
    API_KEY_NAME: str = "X-API-Key"

    STORAGE_CLEANUP_DAYS: int = int(os.getenv("STORAGE_CLEANUP_DAYS", "7"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scenetrace")

# ---- Thread-safe performance benchmark ----
import threading
import time

class BenchmarkStats:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._reset()
        return cls._instance

    def _reset(self):
        self.videos_indexed = 0
        self.total_frames = 0
        self.total_keyframes = 0
        self.total_scan_time = 0.0
        self.total_extract_time = 0.0
        self.total_embed_time = 0.0
        self.total_index_time = 0.0
        self._query_times = []
        self._start = time.time()

    def record_index(self, scan_t, extract_t, embed_t, total_t, total_frames, keyframes):
        with self._lock:
            self.videos_indexed += 1
            self.total_frames += total_frames
            self.total_keyframes += keyframes
            self.total_scan_time += scan_t
            self.total_extract_time += extract_t
            self.total_embed_time += embed_t
            self.total_index_time += total_t

    def record_query(self, seconds: float):
        with self._lock:
            self._query_times.append(seconds)
            if len(self._query_times) > 200:
                self._query_times = self._query_times[-200:]

    def stats(self):
        with self._lock:
            uptime = time.time() - self._start
            q = self._query_times
            avg_q = sum(q) / len(q) if q else 0
            reduction = 0
            if self.total_frames > 0:
                reduction = round((1 - self.total_keyframes / self.total_frames) * 100, 1)
            speed = 0
            if self.total_index_time > 0:
                speed = round(self.total_frames / self.total_index_time, 1)
            return {
                "uptime_seconds": round(uptime),
                "videos_indexed": self.videos_indexed,
                "total_frames": self.total_frames,
                "total_keyframes": self.total_keyframes,
                "avg_frame_reduction_pct": reduction,
                "total_index_time": round(self.total_index_time, 1),
                "avg_scan_time": round(self.total_scan_time / max(self.videos_indexed, 1), 1),
                "avg_extract_time": round(self.total_extract_time / max(self.videos_indexed, 1), 1),
                "avg_embed_time": round(self.total_embed_time / max(self.videos_indexed, 1), 1),
                "avg_index_time": round(self.total_index_time / max(self.videos_indexed, 1), 1),
                "indexing_speed_fps": speed,
                "avg_query_latency": round(avg_q, 3),
                "total_queries": len(q),
            }

benchmark = BenchmarkStats()
