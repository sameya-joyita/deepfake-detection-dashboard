"""End-to-end orchestration for one uploaded video."""

import logging
import threading
import time

from app.models.registry import load_model_bundle
from app.services.evidence import build_evidence_record
from app.services.gradcam import GradCAM
from app.services.inference import InferenceEngine
from app.services.input_quality import assess_input_adequacy
from app.services.media import image_to_data_url
from app.services.preprocessing import FaceExtractor


LOGGER = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, config, bundle):
        self.config = config
        self.bundle = bundle
        self.extractor = FaceExtractor(
            face_detector=bundle.face_detector,
            input_size=config["MODEL_INPUT_SIZE"],
            confidence_threshold=config["FACE_CONFIDENCE_THRESHOLD"],
            margin=config["FACE_MARGIN"],
            jpeg_quality=config["JPEG_QUALITY"],
        )
        self.inference_engine = InferenceEngine(bundle)
        self.gradcam = GradCAM(bundle.dual_model) if config["ENABLE_GRADCAM"] else None
        # OpenCV DNN and PyTorch modules are shared mutable objects. Serialize
        # their use until a process-per-worker deployment is introduced.
        self.analysis_lock = threading.Lock()

    @classmethod
    def from_config(cls, config):
        return cls(config, load_model_bundle(config))

    def health(self):
        return {
            "ready": True,
            "device": str(self.bundle.device),
            "model_version": self.bundle.metadata.get("model_version"),
            "official_checkpoint": self.bundle.metadata.get(
                "official_checkpoint", "dual_best.pth"
            ),
            "gradcam_enabled": bool(self.gradcam is not None),
        }

    def analyse_video(self, video_path, analysis_id, filename, include_gradcam=True):
        pipeline_start = time.perf_counter()

        with self.analysis_lock:
            preprocessing_start = time.perf_counter()
            extraction = self.extractor.extract_video(
                video_path,
                requested_frames=self.config["MAX_VIDEO_FRAMES"],
            )
            preprocessing_ms = (
                time.perf_counter() - preprocessing_start
            ) * 1000

            adequacy = assess_input_adequacy(
                extraction,
                minimum_usable_frames=self.config["MIN_USABLE_FRAMES"],
            )

            inference = None
            gradcam_record = None

            if adequacy["sufficient_evidence"]:
                inference = self.inference_engine.score(
                    [sample.crop_rgb for sample in extraction.samples]
                )

                if include_gradcam and self.gradcam is not None:
                    representative_index = min(
                        range(len(inference.frame_dual_scores)),
                        key=lambda index: abs(
                            inference.frame_dual_scores[index]
                            - inference.dual_score
                        ),
                    )
                    representative = extraction.samples[representative_index]
                    try:
                        overlay, probability, predicted_label = (
                            self.gradcam.generate(
                                representative.crop_rgb,
                                self.bundle.device,
                            )
                        )
                        gradcam_record = {
                            "representative_frame_index": int(
                                representative.frame_index
                            ),
                            "frame_score": float(probability),
                            "explained_predicted_label": int(predicted_label),
                            "overlay_data_url": image_to_data_url(
                                overlay, jpeg_quality=85, max_width=224
                            ),
                            "interpretation": (
                                "Sensitivity of the dual model's spatial "
                                "features for the predicted class; not "
                                "manipulation localisation."
                            ),
                        }
                    except Exception:
                        LOGGER.exception(
                            "Grad-CAM failed; returning the core prediction."
                        )

            frame_previews = self._frame_previews(extraction, inference)

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        processing = {
            "preprocessing_ms": float(preprocessing_ms),
            "total_pipeline_ms": float(total_ms),
            "model_forward_ms": (
                inference.timings_ms if inference is not None else None
            ),
            "device": str(self.bundle.device),
            "note": (
                "Total pipeline time includes video decoding, face detection, "
                "preprocessing, model inference, frame-preview encoding and "
                "optional Grad-CAM; it excludes network transfer."
            ),
        }

        return build_evidence_record(
            analysis_id=analysis_id,
            filename=filename,
            inference=inference,
            adequacy=adequacy,
            frame_previews=frame_previews,
            gradcam_record=gradcam_record,
            processing=processing,
            metadata=self.bundle.metadata,
            dual_checkpoint_sha256=self.bundle.dual_checkpoint_sha256,
            margin_threshold=self.bundle.margin_thresholds[0.90],
            prediction_threshold=self.config["PREDICTION_THRESHOLD"],
        )

    def _frame_previews(self, extraction, inference):
        previews = []
        for index, sample in enumerate(extraction.samples):
            record = {
                "sequence": int(index + 1),
                "video_frame_index": int(sample.frame_index),
                "face_confidence": float(sample.face_confidence),
                "bounding_box": [int(value) for value in sample.bounding_box],
                "face_area_fraction": float(sample.face_area_fraction),
                "preview_data_url": image_to_data_url(
                    sample.preview_rgb, jpeg_quality=75, max_width=480
                ),
                "face_crop_data_url": image_to_data_url(
                    sample.crop_rgb, jpeg_quality=85, max_width=224
                ),
            }
            if inference is not None:
                record.update(
                    {
                        "spatial_score": float(
                            inference.frame_spatial_scores[index]
                        ),
                        "dual_score": float(inference.frame_dual_scores[index]),
                        "frequency_disabled_score": float(
                            inference.frame_frequency_disabled_scores[index]
                        ),
                        "gate_alpha": float(
                            inference.frame_gate_alphas[index]
                        ),
                    }
                )
            previews.append(record)
        return previews
