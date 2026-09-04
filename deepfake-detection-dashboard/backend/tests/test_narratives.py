import unittest

from app.services.narratives import (
    narrate_frequency_counterfactual,
    narrate_gate_alpha,
    narrate_gradcam,
    narrate_model_agreement,
)


class NarrativeTests(unittest.TestCase):
    def test_alpha_is_not_described_as_contribution(self):
        text = narrate_gate_alpha(0.8)
        self.assertIn("weight assigned", text)
        self.assertIn("not by itself a contribution score", text)

    def test_counterfactual_direction(self):
        positive = narrate_frequency_counterfactual(0.2)
        negative = narrate_frequency_counterfactual(-0.2)
        self.assertIn("increased this video's fake score", positive)
        self.assertIn("decreased this video's fake score", negative)

    def test_gradcam_has_required_limitation(self):
        self.assertIn("does not prove", narrate_gradcam(True))

    def test_disagreement_is_a_caution(self):
        text = narrate_model_agreement(False, 0.2, 0.8)
        self.assertIn("different predicted classes", text)
        self.assertIn("does not set the triage rule", text)


if __name__ == "__main__":
    unittest.main()
