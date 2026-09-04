"""Predicted-class Grad-CAM for the dual model's spatial branch."""

import cv2
import numpy as np
import torch

from app.services.inference import crops_to_tensors


class GradCAM:
    def __init__(self, dual_model):
        self.model = dual_model
        self.activations = None
        self.gradients = None
        layer = dual_model.spatial.backbone.blocks[-1]
        self.forward_handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients):
        self.gradients = gradients.detach()

    def generate(self, crop_rgb, device):
        normalized, raw = crops_to_tensors([crop_rgb], device)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        normalized = normalized.detach().requires_grad_(True)
        logits, _ = self.model(normalized, raw)
        probability = float(torch.sigmoid(logits).item())
        predicted_label = 1 if probability >= 0.5 else 0

        target = logits if predicted_label == 1 else -logits
        target.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model features.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu(
            (weights * self.activations).sum(dim=1)
        ).squeeze(0)
        heatmap = heatmap.detach().float().cpu().numpy()
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmap = cv2.resize(heatmap, (crop_rgb.shape[1], crop_rgb.shape[0]))

        colour_map = cv2.cvtColor(
            cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            ),
            cv2.COLOR_BGR2RGB,
        )
        overlay = (0.45 * colour_map + 0.55 * crop_rgb).astype(np.uint8)
        return overlay, probability, predicted_label

    def close(self):
        self.forward_handle.remove()
