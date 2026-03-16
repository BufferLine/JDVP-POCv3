from __future__ import annotations

import unittest

from src.method.tracks.heuristic_baseline import HeuristicBaselineTrack


class HeuristicBaselineTrackTests(unittest.TestCase):
    def test_human_led_decision_support_maps_to_human_active(self) -> None:
        track = HeuristicBaselineTrack()
        output = track.extract(
            interaction_id="session-1",
            turn_number=0,
            human_input="I need help deciding if this trip fits my priorities.",
            ai_response="Tell me what matters most to you.",
            context_turns=[],
            context_module="general",
        )
        self.assertEqual(output.jsv_hint["judgment_holder"], "Human")
        self.assertEqual(output.jsv_hint["delegation_awareness"], "Explicit")
        self.assertEqual(output.jsv_hint["cognitive_engagement"], "Active")
        self.assertEqual(output.jsv_hint["information_seeking"], "Active")

    def test_recommend_request_maps_to_ai_judgment(self) -> None:
        track = HeuristicBaselineTrack()
        output = track.extract(
            interaction_id="session-1",
            turn_number=2,
            human_input="Just recommend one plan for me.",
            ai_response="I recommend option A.",
            context_turns=[],
            context_module="general",
        )
        self.assertEqual(output.jsv_hint["judgment_holder"], "AI")
        self.assertEqual(output.evidence_spans[0]["category"], "delegation_signal")


if __name__ == "__main__":
    unittest.main()
