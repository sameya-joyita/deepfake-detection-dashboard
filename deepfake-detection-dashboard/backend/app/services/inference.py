"""Locked spatial, dual and frequency-disabled video scoring."""

from dataclasses import dataclass
import time

import numpy as np
import torch


IMAGE_NET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGE_NET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def crops_to_tensors(crops_rgb, device):
    if not crops_rgb:
        raise ValueError("At least one face crop is required for inference.")

    raw_images = np.stack(crops_rgb).astype(np.float32) / 255.0
    raw_tensor = torch.from_numpy(raw_images).permute(0, 3, 1, 2)
    mean = IMAGE_NET_MEAN.to(dtype=raw_tensor.dtype)
    std = IMAGE_NET_STD.to(dtype=raw_tensor.dtype)
    normalized_tensor = (raw_tensor - mean) / std
    return normalized_tensor.to(device), raw_tensor.to(device)


@dataclass
class InferenceResult:
    spatial_score: float
    dual_score: float
    frequency_disabled_score: float
    frequency_score_effect: float
    mean_gate_alpha: float
    frame_spatial_scores: list
    frame_dual_scores: list
    frame_frequency_disabled_scores: list
    frame_gate_alphas: list
    timings_ms: dict


class InferenceEngine:
    def __init__(self, model_bundle):
        self.bundle = model_bundle

    def score(self, crops_rgb):
        normalized, raw = crops_to_tensors(crops_rgb, self.bundle.device)

        self.bundle.spatial_model.eval()
        self.bundle.dual_model.eval()

        _synchronise(self.bundle.device)
        start = time.perf_counter()
        with torch.inference_mode():
            spatial_logits = self.bundle.spatial_model(normalized)
        _synchronise(self.bundle.device)
        spatial_ms = (time.perf_counter() - start) * 1000

        _synchronise(self.bundle.device)
        start = time.perf_counter()
        with torch.inference_mode():
            dual_logits, gate_alphas = self.bundle.dual_model(normalized, raw)
        _synchronise(self.bundle.device)
        dual_ms = (time.perf_counter() - start) * 1000

        _synchronise(self.bundle.device)
        start = time.perf_counter()
        with torch.inference_mode():
            disabled_logits = self.bundle.dual_model.forward_frequency_disabled(
                normalized
            )
        _synchronise(self.bundle.device)
        counterfactual_ms = (time.perf_counter() - start) * 1000

        spatial_scores = torch.sigmoid(spatial_logits).detach().float().cpu()
        dual_scores = torch.sigmoid(dual_logits).detach().float().cpu()
        disabled_scores = torch.sigmoid(disabled_logits).detach().float().cpu()
        alphas = gate_alphas.detach().float().view(-1).cpu()

        spatial_video_score = float(spatial_scores.mean())
        dual_video_score = float(dual_scores.mean())
        disabled_video_score = float(disabled_scores.mean())

        return InferenceResult(
            spatial_score=spatial_video_score,
            dual_score=dual_video_score,
            frequency_disabled_score=disabled_video_score,
            frequency_score_effect=(
                dual_video_score - disabled_video_score
            ),
            mean_gate_alpha=float(alphas.mean()),
            frame_spatial_scores=spatial_scores.numpy().tolist(),
            frame_dual_scores=dual_scores.numpy().tolist(),
            frame_frequency_disabled_scores=disabled_scores.numpy().tolist(),
            frame_gate_alphas=alphas.numpy().tolist(),
            timings_ms={
                "spatial_forward_batch": float(spatial_ms),
                "dual_forward_batch": float(dual_ms),
                "frequency_disabled_forward_batch": float(counterfactual_ms),
                "timing_scope": (
                    "Model forward passes only; preprocessing and transfer are "
                    "reported separately in total pipeline time."
                ),
            },
        )
