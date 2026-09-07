from __future__ import annotations

import unittest

from src.method.tracks.heuristic_baseline import HeuristicBaselineTrack


class HeuristicBaselineTrackTests(unittest.TestCase):
    def test_unmatched_turn_leaves_judgment_holder_unobservable(self) -> None:
        output = HeuristicBaselineTrack().extract(
            interaction_id="session-1",
            turn_number=0,
            human_input="Thanks, that is helpful.",
            ai_response="You're welcome.",
            context_turns=[],
            context_module="general",
        )

        self.assertIsNone(output.jsv_hint["judgment_holder"])
        self.assertEqual(output.evidence_spans[0]["category"], "default_excerpt")

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
        self.assertEqual(output.jsv_hint["judgment_holder"], 2)
        self.assertEqual(output.jsv_hint["delegation_awareness"], 2)
        self.assertEqual(output.jsv_hint["cognitive_engagement"], 2)
        self.assertEqual(output.jsv_hint["information_seeking"], 2)

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
        self.assertEqual(output.jsv_hint["judgment_holder"], 9)
        self.assertEqual(output.evidence_spans[0]["category"], "delegation_signal")


if __name__ == "__main__":
    unittest.main()
