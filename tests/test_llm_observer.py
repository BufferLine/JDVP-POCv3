from __future__ import annotations

import unittest

from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response
from src.method.tracks.llm_observer import LLMObserverTrack


VALID_RESPONSE = """
{
  "judgment_holder": "AI",
  "delegation_awareness": "Implicit",
  "cognitive_engagement": "Reactive",
  "information_seeking": "Passive",
  "confidence": {
    "judgment_holder": "medium",
    "delegation_awareness": "medium",
    "cognitive_engagement": "medium",
    "information_seeking": "medium"
  },
  "evidence_spans": [
    {"text": "What would you choose if you were me?", "category": "delegation_signal"}
  ],
  "observer_confidence": 0.77,
  "observer_notes": "Detected direct delegation request"
}
"""


class FakeProvider:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response_text


class LLMNormalizationTests(unittest.TestCase):
    def test_normalize_valid_response(self) -> None:
        normalized = normalize_llm_response(VALID_RESPONSE)
        self.assertEqual(normalized["jsv_hint"]["judgment_holder"], "AI")
        self.assertEqual(normalized["observer_confidence"], 0.77)
        self.assertEqual(len(normalized["evidence_spans"]), 1)

    def test_invalid_enum_is_rejected(self) -> None:
        invalid = VALID_RESPONSE.replace('"AI"', '"Robot"', 1)
        with self.assertRaises(LLMNormalizationError):
            normalize_llm_response(invalid)

    def test_missing_evidence_is_rejected(self) -> None:
        invalid = """
        {
          "judgment_holder": "AI",
          "delegation_awareness": "Implicit",
          "cognitive_engagement": "Reactive",
          "information_seeking": "Passive",
          "confidence": {
            "judgment_holder": "medium",
            "delegation_awareness": "medium",
            "cognitive_engagement": "medium",
            "information_seeking": "medium"
          },
          "evidence_spans": []
        }
        """
        with self.assertRaises(LLMNormalizationError):
            normalize_llm_response(invalid)


class LLMObserverTrackTests(unittest.TestCase):
    def test_llm_track_extracts_track_output(self) -> None:
        track = LLMObserverTrack(provider=FakeProvider(VALID_RESPONSE), model_id="fake-model")
        output = track.extract(
            interaction_id="session-1",
            turn_number=2,
            human_input="What would you choose if you were me?",
            ai_response="I would choose option A.",
            context_turns=[],
            context_module="general",
        )
        self.assertEqual(output.track_id, "llm_observer")
        self.assertEqual(output.jsv_hint["judgment_holder"], "AI")
        self.assertEqual(output.evidence_spans[0]["category"], "delegation_signal")


if __name__ == "__main__":
    unittest.main()
