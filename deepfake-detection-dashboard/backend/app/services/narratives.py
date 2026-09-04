"""Deterministic explanations using only measured backend signals."""


def narrate_gate_alpha(mean_alpha):
    if mean_alpha is None:
        return "Gate alpha was not available for this video."
    position = "above" if mean_alpha >= 0.5 else "below"
    return (
        f"The mean fusion-gate alpha was {mean_alpha:.2f}, {position} the "
        "gate midpoint of 0.50. Alpha is the weight assigned to the frequency "
        "residual; it is not by itself a contribution score."
    )


def narrate_frequency_counterfactual(effect):
    if effect > 0:
        return (
            f"Disabling the frequency residual decreased the manipulation "
            f"score by {effect:.3f}. The residual therefore increased this "
            "video's fake score."
        )
    if effect < 0:
        return (
            f"Disabling the frequency residual increased the manipulation "
            f"score by {abs(effect):.3f}. The residual therefore decreased "
            "this video's fake score."
        )
    return (
        "Disabling the frequency residual produced no score change at the "
        "reported precision."
    )


def narrate_model_agreement(agree, spatial_score, dual_score):
    if agree:
        return (
            f"The spatial score ({spatial_score:.3f}) and dual score "
            f"({dual_score:.3f}) produce the same predicted class."
        )
    return (
        f"The spatial score ({spatial_score:.3f}) and dual score "
        f"({dual_score:.3f}) produce different predicted classes. This "
        "disagreement is shown as a caution, but does not set the triage rule."
    )


def narrate_input_adequacy(adequacy):
    if not adequacy["sufficient_evidence"]:
        return (
            "Insufficient facial evidence was available. No authenticity "
            "assessment has been produced."
        )
    sentence = (
        f"{adequacy['usable_frames']} of {adequacy['requested_frames']} "
        "requested frames provided usable facial evidence."
    )
    if adequacy.get("warnings"):
        sentence += " " + " ".join(adequacy["warnings"])
    return sentence


def narrate_gradcam(available):
    if not available:
        return "No Grad-CAM heatmap was generated for this video."
    return (
        "The Grad-CAM heatmap shows regions that influenced the spatial "
        "features used by the prediction. It does not prove that manipulation "
        "occurred in the highlighted pixels."
    )
