"""Notebook-parity video sampling and face-crop extraction."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.errors import VideoReadError


@dataclass
class FaceSample:
    frame_index: int
    face_confidence: float
    bounding_box: tuple
    face_area_fraction: float
    crop_rgb: np.ndarray
    preview_rgb: np.ndarray


@dataclass
class ExtractionResult:
    total_video_frames: int
    requested_frames: int
    sampled_frame_indices: list
    frames_read: int
    samples: list

    @property
    def face_detection_failures(self):
        return self.frames_read - len(self.samples)


def uniform_frame_indices(total_frames, requested_frames):
    if requested_frames < 1:
        raise ValueError("requested_frames must be at least 1.")
    if total_frames < 1:
        return []
    if total_frames < requested_frames:
        return list(range(total_frames))
    return np.linspace(
        0, total_frames - 1, requested_frames, dtype=int
    ).tolist()


def _count_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoReadError("The uploaded file could not be opened as a video.")

    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count > 0:
        capture.release()
        return count

    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


class FaceExtractor:
    def __init__(
        self,
        face_detector,
        input_size=224,
        confidence_threshold=0.90,
        margin=0.15,
        jpeg_quality=95,
    ):
        self.face_detector = face_detector
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.margin = float(margin)
        self.jpeg_quality = int(jpeg_quality)

    def extract_video(self, video_path, requested_frames=20):
        total_frames = _count_frames(video_path)
        if total_frames < 1:
            raise VideoReadError("No decodable frames were found in the video.")

        indices = uniform_frame_indices(total_frames, requested_frames)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoReadError("The uploaded video could not be reopened.")

        samples = []
        frames_read = 0

        try:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    continue

                frames_read += 1
                sample = self._detect_and_crop(frame_bgr, frame_index)
                if sample is not None:
                    samples.append(sample)
        finally:
            capture.release()

        return ExtractionResult(
            total_video_frames=total_frames,
            requested_frames=int(requested_frames),
            sampled_frame_indices=indices,
            frames_read=frames_read,
            samples=samples,
        )

    def _detect_and_crop(self, frame_bgr, frame_index):
        height, width = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (300, 300))
        blob = cv2.dnn.blobFromImage(
            resized,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        best_box = None
        best_confidence = 0.0

        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if (
                confidence < self.confidence_threshold
                or confidence <= best_confidence
            ):
                continue

            x1 = int(detections[0, 0, index, 3] * width)
            y1 = int(detections[0, 0, index, 4] * height)
            x2 = int(detections[0, 0, index, 5] * width)
            y2 = int(detections[0, 0, index, 6] * height)
            best_box = (x1, y1, x2, y2)
            best_confidence = confidence

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box
        pad_x = int((x2 - x1) * self.margin)
        pad_y = int((y2 - y1) * self.margin)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop_bgr = cv2.resize(
            frame_bgr[y1:y2, x1:x2],
            (self.input_size, self.input_size),
        )

        # Research preprocessing wrote every crop as JPEG quality 95 before
        # evaluation. Reproduce that round trip in memory for live parity.
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            crop_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            return None
        crop_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            return None

        preview_bgr = frame_bgr.copy()
        cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

        face_area_fraction = float(
            ((x2 - x1) * (y2 - y1)) / (width * height)
        )

        return FaceSample(
            frame_index=int(frame_index),
            face_confidence=float(best_confidence),
            bounding_box=(int(x1), int(y1), int(x2), int(y2)),
            face_area_fraction=face_area_fraction,
            crop_rgb=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
            preview_rgb=cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB),
        )
