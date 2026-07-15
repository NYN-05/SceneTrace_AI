import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

def test_defaults():
    assert settings.CORS_ORIGINS == "http://localhost:5173"
    assert settings.MAX_UPLOAD_MB == 500
    assert settings.RATE_LIMIT is not None

def test_env_override(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://example.com")
    monkeypatch.setenv("MAX_UPLOAD_MB", "100")
    import importlib
    import config
    importlib.reload(config)
    assert config.settings.CORS_ORIGINS == "http://example.com"
    assert config.settings.MAX_UPLOAD_MB == 100
