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
