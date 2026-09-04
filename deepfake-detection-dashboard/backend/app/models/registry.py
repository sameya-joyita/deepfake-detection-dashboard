"""Strict loading of the trusted detector and frozen checkpoints."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import math
import torch

from app.errors import ArtifactError
from app.models.architectures import DualBranchFinal, SpatialBranch


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _trusted_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _state_dict_from_checkpoint(checkpoint, path):
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ArtifactError(
            f"Checkpoint does not contain model_state: {path}"
        )
    return checkpoint["model_state"]


def select_device(requested):
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ArtifactError("DEVICE=cuda was requested but CUDA is unavailable.")
    if requested not in {"cpu", "cuda"}:
        raise ArtifactError("DEVICE must be auto, cpu or cuda.")
    return torch.device(requested)


@dataclass
class ModelBundle:
    spatial_model: SpatialBranch
    dual_model: DualBranchFinal
    face_detector: object
    device: torch.device
    metadata: dict
    margin_thresholds: dict
    dual_checkpoint_sha256: str


def load_model_bundle(config):
    required_paths = [
        Path(config["SPATIAL_CHECKPOINT"]),
        Path(config["DUAL_CHECKPOINT"]),
        Path(config["FACE_PROTOTXT"]),
        Path(config["FACE_WEIGHTS"]),
        Path(config["TRIAGE_THRESHOLDS"]),
        Path(config["MODEL_METADATA"]),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ArtifactError("Required artifacts are missing: " + ", ".join(missing))

    with open(config["MODEL_METADATA"], encoding="utf-8") as source:
        metadata = json.load(source)
    with open(config["TRIAGE_THRESHOLDS"], encoding="utf-8") as source:
        threshold_record = json.load(source)

    margin_thresholds = {
        float(key): float(value)
        for key, value in threshold_record["margin_thresholds"].items()
    }
    if 0.90 not in margin_thresholds:
        raise ArtifactError("The locked 90% coverage threshold is missing.")
    margin_threshold = margin_thresholds[0.90]
    if not math.isfinite(margin_threshold) or not 0 <= margin_threshold <= 0.5:
        raise ArtifactError("The locked 90% coverage threshold is invalid.")

    dual_hash = sha256_file(config["DUAL_CHECKPOINT"])
    expected_hash = metadata.get("official_checkpoint_sha256")
    if config["VERIFY_DUAL_SHA256"]:
        if not expected_hash:
            raise ArtifactError(
                "model_metadata.json does not identify the official checkpoint."
            )
        if dual_hash.lower() != str(expected_hash).lower():
            raise ArtifactError(
                "dual_best.pth does not match the official checkpoint SHA-256."
            )

    device = select_device(config["DEVICE"])

    spatial_model = SpatialBranch(pretrained=False, dropout=0.4).to(device)
    spatial_checkpoint = _trusted_torch_load(
        config["SPATIAL_CHECKPOINT"], device
    )
    spatial_model.load_state_dict(
        _state_dict_from_checkpoint(
            spatial_checkpoint, config["SPATIAL_CHECKPOINT"]
        ),
        strict=True,
    )
    spatial_model.eval()

    dual_model = DualBranchFinal(dropout=0.4).to(device)
    dual_checkpoint = _trusted_torch_load(config["DUAL_CHECKPOINT"], device)
    dual_model.load_state_dict(
        _state_dict_from_checkpoint(dual_checkpoint, config["DUAL_CHECKPOINT"]),
        strict=True,
    )
    dual_model.eval()

    face_detector = cv2.dnn.readNetFromCaffe(
        str(config["FACE_PROTOTXT"]), str(config["FACE_WEIGHTS"])
    )
    face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return ModelBundle(
        spatial_model=spatial_model,
        dual_model=dual_model,
        face_detector=face_detector,
        device=device,
        metadata=metadata,
        margin_thresholds=margin_thresholds,
        dual_checkpoint_sha256=dual_hash,
    )
