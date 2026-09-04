import unittest
from types import SimpleNamespace

from app.services.evidence import (
    build_evidence_record,
    predicted_label,
    triage_outcome,
)


class EvidenceTests(unittest.TestCase):
    def test_prediction_threshold(self):
        self.assertEqual(predicted_label(0.4999), 0)
        self.assertEqual(predicted_label(0.5), 1)

    def test_insufficient_evidence_has_priority(self):
        outcome = triage_outcome(0.99, 0.29, False)
        self.assertEqual(outcome, "unable_to_assess")

    def test_low_margin_is_referred(self):
        outcome = triage_outcome(0.6, 0.29, True)
        self.assertEqual(outcome, "inconclusive_manual_review")

    def test_high_margin_is_labelled(self):
        self.assertEqual(triage_outcome(0.9, 0.29, True), "likely_manipulated")
        self.assertEqual(triage_outcome(0.1, 0.29, True), "likely_authentic")

    def test_live_record_has_no_invented_combined_score(self):
        inference = SimpleNamespace(
            spatial_score=0.7,
            dual_score=0.9,
            frequency_disabled_score=0.4,
            frequency_score_effect=0.5,
            mean_gate_alpha=0.8,
            frame_dual_scores=[0.8, 1.0],
        )
        adequacy = {
            "requested_frames": 20,
            "usable_frames": 20,
            "sufficient_evidence": True,
            "warnings": [],
        }
        record = build_evidence_record(
            analysis_id="test",
            filename="video.mp4",
            inference=inference,
            adequacy=adequacy,
            frame_previews=[],
            gradcam_record=None,
            processing={},
            metadata={
                "model_version": "test",
                "benchmarks": {
                    "ffpp_source_disjoint_test": {
                        "spatial_video_auc": 0.9884,
                        "dual_video_auc": 0.9901,
                    }
                },
            },
            dual_checkpoint_sha256="abc",
            margin_threshold=0.29,
        )
        self.assertNotIn("combined_score", record["scores"])
        self.assertEqual(record["decision"]["outcome"], "likely_manipulated")
        self.assertFalse(record["model"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
