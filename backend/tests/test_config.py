import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

def test_defaults():
    assert settings.CORS_ORIGINS == "http://localhost:5173"
    assert settings.MAX_UPLOAD_MB == 500
    assert settings.RATE_LIMIT is not None
    assert settings.CLIP_MODEL_NAME == "openai/clip-vit-base-patch32"
    assert settings.DETECTOR_MODEL_NAME == "IDEA-Research/grounding-dino-base"
    assert settings.SEMANTIC_WEIGHT == 0.55
    assert settings.OBJECT_WEIGHT == 0.45
    assert settings.DETECTION_THRESHOLD == 0.2
    assert settings.GAP_THRESHOLD == 3
    assert settings.MOTION_STRIDE == 3
    assert settings.BATCH_SIZE == 64
    assert settings.NUM_WORKERS == 4
    assert settings.DET_CACHE_MAXSIZE == 5000
    assert settings.DET_CACHE_TTL == 3600
    assert settings.STORAGE_CLEANUP_DAYS == 7
    assert settings.REQUEST_TIMEOUT_SECONDS == 120

def test_env_override(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://example.com")
    monkeypatch.setenv("MAX_UPLOAD_MB", "100")
    monkeypatch.setenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
    monkeypatch.setenv("DET_CACHE_MAXSIZE", "1000")
    import importlib
    import config
    importlib.reload(config)
    assert config.settings.CORS_ORIGINS == "http://example.com"
    assert config.settings.MAX_UPLOAD_MB == 100
    assert config.settings.CLIP_MODEL_NAME == "openai/clip-vit-large-patch14"
    assert config.settings.DET_CACHE_MAXSIZE == 1000
