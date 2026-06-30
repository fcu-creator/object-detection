from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

WEB_HOST = os.environ.get("TOOLKNIFE_WEB_HOST", os.environ.get("TOOLKNIFE_HOST", "0.0.0.0"))
WEB_PORT = int(os.environ.get("TOOLKNIFE_WEB_PORT", os.environ.get("TOOLKNIFE_PORT", "8001")))
DETECTION_HOST = os.environ.get("TOOLKNIFE_DETECTION_HOST", "0.0.0.0")
DETECTION_PORT = int(os.environ.get("TOOLKNIFE_DETECTION_PORT", "8002"))
KEYPOINT_HOST = os.environ.get("TOOLKNIFE_KEYPOINT_HOST", "0.0.0.0")
KEYPOINT_PORT = int(os.environ.get("TOOLKNIFE_KEYPOINT_PORT", "8003"))
CROP_HOST = os.environ.get("TOOLKNIFE_CROP_HOST", "0.0.0.0")
CROP_PORT = int(os.environ.get("TOOLKNIFE_CROP_PORT", "8004"))
HOST_REAL_IP = os.environ.get("HOST_REAL_IP", "").strip()


def service_call_host(host: str) -> str:
    if host in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return host

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = DATA_DIR / "logs"
DATASET_DIR = DATA_DIR / "datasets"
TRAINING_RUN_DIR = DATA_DIR / "training_runs"
TMP_DIR = DATA_DIR / "tmp"
ULTRALYTICS_CONFIG_DIR = DATA_DIR / "ultralytics_config"

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

WEIGHTS_DIR = ROOT / "weights"
DETECTION_WEIGHT = WEIGHTS_DIR / "detection_best.pt"
KEYPOINT_WEIGHT = WEIGHTS_DIR / "keypoint_best.pt"
ACCESS_TOKEN_FILE = CONFIG_DIR / "access_token.txt"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640


def ensure_directories() -> None:
    for folder in (
        CONFIG_DIR,
        UPLOAD_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        DATASET_DIR,
        TRAINING_RUN_DIR,
        TMP_DIR,
        ULTRALYTICS_CONFIG_DIR,
        WEIGHTS_DIR,
        OUTPUT_DIR / "det",
        OUTPUT_DIR / "keypoint",
        OUTPUT_DIR / "crop",
        OUTPUT_DIR / "pipeline",
    ):
        folder.mkdir(parents=True, exist_ok=True)


def ensure_weight(path: Path, label: str) -> None:
    ensure_directories()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} model weight: {path}")
