"""Architectures copied from the final, validated research notebook."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SpatialBranch(nn.Module):
    """EfficientNet-B4 spatial feature extractor and binary logit head."""

    def __init__(self, pretrained=False, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 1),
        )

    def forward(self, normalized_images):
        features = self.backbone(normalized_images)
        return self.classifier(features).squeeze(1)

    def get_features(self, normalized_images):
        return self.backbone(normalized_images)


class FrequencyBranchV2(nn.Module):
    """2D FFT magnitude branch operating on raw RGB tensors in `[0,1]`."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, raw_images):
        gray = (
            0.299 * raw_images[:, 0]
            + 0.587 * raw_images[:, 1]
            + 0.114 * raw_images[:, 2]
        )
        spectrum = torch.fft.fft2(gray, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        magnitude = torch.abs(spectrum)

        if magnitude.shape != gray.shape:
            raise RuntimeError("FFT shift changed the batch dimension order.")

        log_magnitude = torch.log1p(magnitude)
        batch_size = log_magnitude.size(0)
        minimum = log_magnitude.view(batch_size, -1).min(1).values.view(
            batch_size, 1, 1
        )
        maximum = log_magnitude.view(batch_size, -1).max(1).values.view(
            batch_size, 1, 1
        )
        log_magnitude = (log_magnitude - minimum) / (
            maximum - minimum + 1e-8
        )
        log_magnitude = F.interpolate(
            log_magnitude.unsqueeze(1),
            size=(112, 112),
            mode="bilinear",
            align_corners=False,
        )
        return self.cnn(log_magnitude).squeeze(-1).squeeze(-1)


class DualBranchFinal(nn.Module):
    """Gated spatial-frequency residual fusion used by `dual_best.pth`."""

    def __init__(self, dropout=0.4):
        super().__init__()
        self.spatial = SpatialBranch(pretrained=False, dropout=dropout)
        self.freq = FrequencyBranchV2()
        self.freq_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1792),
        )
        self.gate = nn.Sequential(
            nn.Linear(1792 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, normalized_images, raw_images):
        spatial_features = self.spatial.get_features(normalized_images)
        frequency_features = self.freq(raw_images)
        gate_alpha = self.gate(
            torch.cat([spatial_features, frequency_features], dim=1)
        )
        blend = spatial_features + gate_alpha * self.freq_proj(
            frequency_features
        )
        logits = self.classifier(blend).squeeze(1)
        return logits, gate_alpha

    def forward_frequency_disabled(self, normalized_images):
        """Strict counterfactual: classify spatial features with this head."""
        spatial_features = self.spatial.get_features(normalized_images)
        return self.classifier(spatial_features).squeeze(1)
