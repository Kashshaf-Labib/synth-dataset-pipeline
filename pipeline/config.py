"""
Configuration module — loads environment variables and exposes pipeline settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ── Environment detection ────────────────────────────────────────────────────
IS_KAGGLE: bool = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# ── API keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Vertex AI settings ──────────────────────────────────────────────────────
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
GCP_IMAGE_LOCATION: str = os.getenv("GCP_IMAGE_LOCATION", "global")
VERTEX_ENDPOINT_ID: str = os.getenv("VERTEX_ENDPOINT_ID", "")

# ── Model settings ───────────────────────────────────────────────────────────
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
IMAGE_GENERATOR: str = os.getenv("IMAGE_GENERATOR", "gemini")  # "dalle" or "gemini"
GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
DALLE_MODEL: str = os.getenv("DALLE_MODEL", "dall-e-3")
DALLE_IMAGE_SIZE: str = os.getenv("DALLE_IMAGE_SIZE", "1024x1024")
DALLE_IMAGE_QUALITY: str = os.getenv("DALLE_IMAGE_QUALITY", "standard")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# Ensure output dirs exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
