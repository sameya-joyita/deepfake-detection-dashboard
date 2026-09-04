"""Check required paths and the official dual checkpoint hash."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    metadata_path = ROOT / "artifacts" / "model_metadata.json"
    with open(metadata_path, encoding="utf-8") as source:
        metadata = json.load(source)

    paths = {
        "spatial checkpoint": ROOT / "artifacts/checkpoints/spatial_best.pth",
        "dual checkpoint": ROOT / "artifacts/checkpoints/dual_best.pth",
        "face prototxt": ROOT / "artifacts/face_detector/deploy.prototxt",
        "face weights": (
            ROOT
            / "artifacts/face_detector/"
            / "res10_300x300_ssd_iter_140000.caffemodel"
        ),
        "triage thresholds": ROOT / "artifacts/triage_thresholds.json",
    }

    failed = False
    for label, path in paths.items():
        present = path.is_file()
        print(f"{label:<22}: {'present' if present else 'MISSING'} - {path}")
        failed = failed or not present

    dual_path = paths["dual checkpoint"]
    if dual_path.is_file():
        actual = sha256_file(dual_path)
        expected = metadata["official_checkpoint_sha256"]
        matches = actual.lower() == expected.lower()
        print(f"dual SHA-256 match    : {matches}")
        if not matches:
            print(f"  expected: {expected}")
            print(f"  actual  : {actual}")
            failed = True

    if failed:
        sys.exit(1)
    print("Artifact verification passed.")


if __name__ == "__main__":
    main()
