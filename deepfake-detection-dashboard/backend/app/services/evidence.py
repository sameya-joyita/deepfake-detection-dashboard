"""Construction of the live, exportable video evidence record."""

from datetime import datetime, timezone

import numpy as np

from app.services.narratives import (
    narrate_frequency_counterfactual,
    narrate_gate_alpha,
    narrate_gradcam,
    narrate_input_adequacy,
    narrate_model_agreement,
)


DISCLAIMER = (
    "This is a research-prototype output, not proof of manipulation and not "
    "a calibrated probability. Human review is required before any "
    "consequential use."
)


def _frame_statistics(frame_scores):
    values = np.asarray(frame_scores, dtype=float)
    if values.size < 1:
        raise ValueError("Frame statistics require at least one score.")
    return {
        "n_frames": int(values.size),
        "frame_score_mean": float(values.mean()),
        "frame_score_std": (
            float(values.std(ddof=1)) if values.size > 1 else None
        ),
        "frame_score_min": float(values.min()),
        "frame_score_max": float(values.max()),
    }


def predicted_label(score, threshold=0.5):
    return 1 if float(score) >= float(threshold) else 0


def triage_outcome(dual_score, margin_threshold, sufficient_evidence):
    if not sufficient_evidence:
        return "unable_to_assess"
    margin = abs(float(dual_score) - 0.5)
    if margin < float(margin_threshold):
        return "inconclusive_manual_review"
    return "likely_manipulated" if dual_score >= 0.5 else "likely_authentic"


def _decision_record(inference, adequacy, prediction_threshold, margin_threshold):
    if inference is None:
        return {
            "outcome": "unable_to_assess",
            "predicted_label": None,
            "dual_margin": None,
            "locked_90_coverage_margin_threshold": float(margin_threshold),
            "reason": "Insufficient usable facial evidence.",
        }

    outcome = triage_outcome(
        inference.dual_score,
        margin_threshold,
        adequacy["sufficient_evidence"],
    )
    return {
        "outcome": outcome,
        "predicted_label": (
            predicted_label(inference.dual_score, prediction_threshold)
            if outcome not in {"unable_to_assess", "inconclusive_manual_review"}
            else None
        ),
        "dual_margin": float(abs(inference.dual_score - 0.5)),
        "locked_90_coverage_margin_threshold": float(margin_threshold),
        "reason": (
            "The dual-score margin is below the validation-locked operating "
            "threshold."
            if outcome == "inconclusive_manual_review"
            else None
        ),
    }


def build_evidence_record(
    analysis_id,
    filename,
    inference,
    adequacy,
    frame_previews,
    gradcam_record,
    processing,
    metadata,
    dual_checkpoint_sha256,
    margin_threshold,
    prediction_threshold=0.5,
):
    decision = _decision_record(
        inference,
        adequacy,
        prediction_threshold,
        margin_threshold,
    )

    base = {
        "record_type": "live_video_evidence",
        "analysis_id": str(analysis_id),
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source": {"filename": filename},
        "status": decision["outcome"],
        "model": {
            "version": metadata.get("model_version", "video_auc_v2"),
            "official_checkpoint": metadata.get(
                "official_checkpoint", "dual_best.pth"
            ),
            "checkpoint_sha256": dual_checkpoint_sha256,
            "score_definition": metadata.get(
                "score_definition", "mean of per-frame sigmoid probabilities"
            ),
            "calibrated": bool(metadata.get("calibrated", False)),
        },
        "decision": decision,
        "input_adequacy": adequacy,
        "frames": frame_previews,
        "processing": processing,
        "benchmark_context": metadata.get("benchmarks", {}),
        "limitations": metadata.get("limitations", []),
        "disclaimer": DISCLAIMER,
    }

    if inference is None:
        base.update(
            {
                "scores": None,
                "model_comparison": None,
                "frequency_counterfactual": None,
                "frame_evidence": None,
                "gate_alpha": None,
                "explainability": {
                    "gradcam_available": False,
                    "gradcam": None,
                    "mc_dropout_available": False,
                    "mc_dropout": None,
                },
                "narrative": {
                    "input_adequacy": narrate_input_adequacy(adequacy),
                    "gradcam": narrate_gradcam(False),
                },
            }
        )
        return base

    spatial_label = predicted_label(
        inference.spatial_score, prediction_threshold
    )
    dual_label = predicted_label(inference.dual_score, prediction_threshold)
    model_agreement = spatial_label == dual_label
    statistics = _frame_statistics(inference.frame_dual_scores)

    base.update(
        {
            "scores": {
                "spatial_score": float(inference.spatial_score),
                "dual_score": float(inference.dual_score),
                "frequency_disabled_score": float(
                    inference.frequency_disabled_score
                ),
                "frequency_score_effect": float(
                    inference.frequency_score_effect
                ),
            },
            "model_comparison": {
                "spatial": {
                    "video_score": float(inference.spatial_score),
                    "predicted_label": spatial_label,
                    "benchmark_video_auc": metadata.get("benchmarks", {})
                    .get("ffpp_source_disjoint_test", {})
                    .get("spatial_video_auc"),
                },
                "dual": {
                    "video_score": float(inference.dual_score),
                    "predicted_label": dual_label,
                    "benchmark_video_auc": metadata.get("benchmarks", {})
                    .get("ffpp_source_disjoint_test", {})
                    .get("dual_video_auc"),
                },
                "agreement": bool(model_agreement),
                "note": (
                    "Video scores describe this upload. Benchmark AUC values "
                    "describe ranking performance across labelled datasets."
                ),
            },
            "frequency_counterfactual": {
                "frequency_disabled_score": float(
                    inference.frequency_disabled_score
                ),
                "score_effect": float(inference.frequency_score_effect),
                "interpretation_scope": (
                    "Same frozen dual checkpoint with alpha * freq_proj(f) "
                    "removed; mechanistic evidence, not standalone causal proof."
                ),
            },
            "frame_evidence": statistics,
            "gate_alpha": float(inference.mean_gate_alpha),
            "explainability": {
                "gradcam_available": bool(gradcam_record is not None),
                "gradcam": gradcam_record,
                "mc_dropout_available": False,
                "mc_dropout": None,
                "mc_dropout_note": (
                    "MC Dropout remained exploratory in the research notebook "
                    "and does not drive live decisions."
                ),
            },
            "narrative": {
                "gate_alpha": narrate_gate_alpha(inference.mean_gate_alpha),
                "frequency_counterfactual": narrate_frequency_counterfactual(
                    inference.frequency_score_effect
                ),
                "model_agreement": narrate_model_agreement(
                    model_agreement,
                    inference.spatial_score,
                    inference.dual_score,
                ),
                "input_adequacy": narrate_input_adequacy(adequacy),
                "gradcam": narrate_gradcam(gradcam_record is not None),
            },
        }
    )
    return _native_types(base)


def _native_types(value):
    if isinstance(value, dict):
        return {key: _native_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native_types(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
