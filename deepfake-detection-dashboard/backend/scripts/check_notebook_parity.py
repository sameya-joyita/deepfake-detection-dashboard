"""Compare saved FF++ crops with the final notebook reference prediction."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config
from app.models.registry import load_model_bundle
from app.services.inference import InferenceEngine


REFERENCE_VIDEO_ID = "Deepfakes_004_982"
REFERENCE_DUAL_SCORE = 0.935472482442856
REFERENCE_DISABLED_SCORE = 0.0230944835580885
TOLERANCE = 1e-4


def config_dict():
    return {
        name: getattr(Config, name)
        for name in [
            "SPATIAL_CHECKPOINT",
            "DUAL_CHECKPOINT",
            "FACE_PROTOTXT",
            "FACE_WEIGHTS",
            "TRIAGE_THRESHOLDS",
            "MODEL_METADATA",
            "VERIFY_DUAL_SHA256",
            "DEVICE",
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "crop_directory",
        type=Path,
        help=(
            "Directory containing the saved test crops for "
            f"{REFERENCE_VIDEO_ID}."
        ),
    )
    args = parser.parse_args()

    crop_paths = sorted(
        path
        for path in args.crop_directory.glob(f"{REFERENCE_VIDEO_ID}_f*.jpg")
    )
    if not crop_paths:
        raise FileNotFoundError("No notebook reference crops were found.")
    if len(crop_paths) > 20:
        selected = np.linspace(0, len(crop_paths) - 1, 20, dtype=int)
        crop_paths = [crop_paths[index] for index in selected]

    crops = []
    for path in crop_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        crops.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    engine = InferenceEngine(load_model_bundle(config_dict()))
    result = engine.score(crops)

    checks = {
        "dual score": (result.dual_score, REFERENCE_DUAL_SCORE),
        "frequency-disabled score": (
            result.frequency_disabled_score,
            REFERENCE_DISABLED_SCORE,
        ),
    }
    for label, (actual, expected) in checks.items():
        difference = abs(actual - expected)
        print(
            f"{label:<28}: actual={actual:.9f}, expected={expected:.9f}, "
            f"difference={difference:.9f}"
        )
        if difference > TOLERANCE:
            raise RuntimeError(f"Notebook parity failed for {label}.")

    print("Notebook parity passed.")


if __name__ == "__main__":
    main()
