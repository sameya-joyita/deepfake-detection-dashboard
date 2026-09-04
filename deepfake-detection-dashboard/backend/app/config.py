"""Environment-backed Flask configuration."""

import os
import math
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")


def _path_from_env(name, default):
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else BACKEND_ROOT / value


def _as_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
    FRONTEND_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    ]

    DEVICE = os.getenv("DEVICE", "auto")
    MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "20"))
    MIN_USABLE_FRAMES = int(os.getenv("MIN_USABLE_FRAMES", "5"))
    MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "224"))
    FACE_CONFIDENCE_THRESHOLD = float(
        os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.90")
    )
    FACE_MARGIN = float(os.getenv("FACE_MARGIN", "0.15"))
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "95"))
    PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    ENABLE_GRADCAM = _as_bool("ENABLE_GRADCAM", True)
    VERIFY_DUAL_SHA256 = _as_bool("VERIFY_DUAL_SHA256", True)
    LOAD_MODELS = True

    MODEL_DIR = _path_from_env("MODEL_DIR", "artifacts/checkpoints")
    SPATIAL_CHECKPOINT = MODEL_DIR / os.getenv(
        "SPATIAL_CHECKPOINT", "spatial_best.pth"
    )
    DUAL_CHECKPOINT = MODEL_DIR / os.getenv(
        "DUAL_CHECKPOINT", "dual_best.pth"
    )
    FACE_DETECTOR_DIR = _path_from_env(
        "FACE_DETECTOR_DIR", "artifacts/face_detector"
    )
    FACE_PROTOTXT = FACE_DETECTOR_DIR / "deploy.prototxt"
    FACE_WEIGHTS = (
        FACE_DETECTOR_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    )
    TRIAGE_THRESHOLDS = _path_from_env(
        "TRIAGE_THRESHOLDS", "artifacts/triage_thresholds.json"
    )
    MODEL_METADATA = _path_from_env(
        "MODEL_METADATA", "artifacts/model_metadata.json"
    )
    UPLOAD_ROOT = BACKEND_ROOT / "uploads"

    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def validate_config(config):
    if config["MAX_VIDEO_FRAMES"] < 1:
        raise ValueError("MAX_VIDEO_FRAMES must be at least 1.")
    if config["MIN_USABLE_FRAMES"] < 1:
        raise ValueError("MIN_USABLE_FRAMES must be at least 1.")
    if config["MIN_USABLE_FRAMES"] > config["MAX_VIDEO_FRAMES"]:
        raise ValueError("MIN_USABLE_FRAMES cannot exceed MAX_VIDEO_FRAMES.")
    if not 0 < config["FACE_CONFIDENCE_THRESHOLD"] <= 1:
        raise ValueError("FACE_CONFIDENCE_THRESHOLD must be in (0, 1].")
    if not 0 <= config["FACE_MARGIN"] <= 1:
        raise ValueError("FACE_MARGIN must be in [0, 1].")
    if not 1 <= config["JPEG_QUALITY"] <= 100:
        raise ValueError("JPEG_QUALITY must be in [1, 100].")
    if not 0 < config["PREDICTION_THRESHOLD"] < 1:
        raise ValueError("PREDICTION_THRESHOLD must be in (0, 1).")

    locked_values = {
        "MAX_VIDEO_FRAMES": 20,
        "MIN_USABLE_FRAMES": 5,
        "MODEL_INPUT_SIZE": 224,
        "JPEG_QUALITY": 95,
    }
    for name, expected in locked_values.items():
        if config[name] != expected:
            raise ValueError(
                f"{name} is locked to {expected} by the validated protocol."
            )

    locked_floats = {
        "FACE_CONFIDENCE_THRESHOLD": 0.90,
        "FACE_MARGIN": 0.15,
        "PREDICTION_THRESHOLD": 0.5,
    }
    for name, expected in locked_floats.items():
        if not math.isclose(config[name], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"{name} is locked to {expected} by the validated protocol."
            )
