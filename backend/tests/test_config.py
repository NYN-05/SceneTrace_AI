import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, PROJECT_ROOT

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

    expected_root = Path(__file__).parent.parent.parent.resolve()

    assert PROJECT_ROOT == expected_root

    assert settings.LOG_FILE == str(expected_root / "server.log")

    assert settings.CLIP_WEIGHT == 0.20
    assert settings.CAPTION_WEIGHT == 0.20
    assert settings.OBJECT_MATCH_WEIGHT == 0.25
    assert settings.MOTION_MATCH_WEIGHT == 0.10
    assert settings.TRACK_CONSISTENCY_WEIGHT == 0.15
    assert settings.TEMPORAL_WEIGHT == 0.10
    assert settings.RELATIONSHIP_WEIGHT == 0.10

    assert settings.DETECTOR_BACKEND == "yolo_world"
    assert settings.YOLO_WORLD_DEFAULT == "large"
    assert settings.YOLO_WORLD_FALLBACKS == "large,medium,small"

    assert settings.TRACK_MATCH_THRESHOLD == 0.5
    assert settings.TRACK_BUFFER == 30

    assert settings.CAPTIONER_MODEL == "microsoft/Florence-2-base"
    assert settings.CAPTIONER_ENABLED is True
    assert settings.QUERY_PARSER_MODE == "regex"

    assert settings.CLIP_WINDOW_SIZE == 3
    assert settings.CLIP_STRIDE == 2
    assert settings.CLIP_MOTION_WEIGHT == 0.3

    assert settings.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert settings.RERANKER_ENABLED is False

    assert settings.API_KEY == ""
    assert settings.API_KEY_NAME == "X-API-Key"

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
