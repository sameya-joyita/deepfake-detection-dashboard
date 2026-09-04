"""Descriptive input-adequacy checks aligned with the notebook."""

import cv2
import numpy as np


BLUR_WARNING_THRESHOLD = 50.0
DARK_WARNING_THRESHOLD = 40.0
BRIGHT_WARNING_THRESHOLD = 220.0


def assess_input_adequacy(extraction, minimum_usable_frames):
    blur_scores = []
    brightness_values = []
    face_areas = []

    for sample in extraction.samples:
        gray = cv2.cvtColor(sample.crop_rgb, cv2.COLOR_RGB2GRAY)
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_values.append(float(gray.mean()))
        face_areas.append(float(sample.face_area_fraction))

    usable_frames = len(extraction.samples)
    sufficient_evidence = usable_frames >= int(minimum_usable_frames)
    warnings = []

    if extraction.frames_read < len(extraction.sampled_frame_indices):
        warnings.append(
            f"{len(extraction.sampled_frame_indices)-extraction.frames_read} "
            "sampled frames could not be decoded."
        )

    if not sufficient_evidence:
        warnings.append(
            f"Fewer than {minimum_usable_frames} usable face crops were found; "
            "insufficient facial evidence for an authenticity assessment."
        )

    mean_blur = float(np.mean(blur_scores)) if blur_scores else None
    mean_brightness = (
        float(np.mean(brightness_values)) if brightness_values else None
    )
    mean_face_area = float(np.mean(face_areas)) if face_areas else None

    if mean_blur is not None and mean_blur < BLUR_WARNING_THRESHOLD:
        warnings.append("Low sharpness was detected by a display-only heuristic.")

    if mean_brightness is not None and (
        mean_brightness < DARK_WARNING_THRESHOLD
        or mean_brightness > BRIGHT_WARNING_THRESHOLD
    ):
        warnings.append(
            "Extreme brightness was detected by a display-only heuristic."
        )

    return {
        "requested_frames": int(extraction.requested_frames),
        "total_video_frames": int(extraction.total_video_frames),
        "sampled_frames": int(len(extraction.sampled_frame_indices)),
        "frames_read": int(extraction.frames_read),
        "usable_frames": int(usable_frames),
        "face_detection_failures": int(extraction.face_detection_failures),
        "usable_fraction": float(
            usable_frames / extraction.requested_frames
        ),
        "sufficient_evidence": bool(sufficient_evidence),
        "mean_blur_variance": mean_blur,
        "mean_brightness": mean_brightness,
        "mean_face_area_fraction": mean_face_area,
        "quality_threshold_status": (
            "Blur and brightness boundaries are descriptive heuristics; "
            "they do not control triage."
        ),
        "warnings": warnings,
    }
