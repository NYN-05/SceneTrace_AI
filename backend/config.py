import os
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

settings = Settings()
