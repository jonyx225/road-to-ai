"""Application settings, read from environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where the trained model lives. Override with the MODEL_PATH env var.
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "sentiment.joblib"))

# Input limits, enforced by the request schemas.
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
