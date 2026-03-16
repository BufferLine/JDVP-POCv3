from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response
from src.method.tracks.llm_observer import LLMObserverTrack, StaticResponseProvider, create_env_backed_llm_track


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

    def test_env_backed_track_supports_static_response_provider(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            response_path = Path(tmp_dir) / "response.json"
            response_path.write_text(VALID_RESPONSE, encoding="utf-8")
            previous_provider = __import__("os").environ.get("JDVP_LLM_PROVIDER")
            previous_model = __import__("os").environ.get("JDVP_LLM_MODEL")
            previous_path = __import__("os").environ.get("JDVP_LLM_STATIC_RESPONSE_PATH")
            try:
                __import__("os").environ["JDVP_LLM_PROVIDER"] = "static_response"
                __import__("os").environ["JDVP_LLM_MODEL"] = "static-model"
                __import__("os").environ["JDVP_LLM_STATIC_RESPONSE_PATH"] = str(response_path)
                track = create_env_backed_llm_track()
            finally:
                if previous_provider is None:
                    __import__("os").environ.pop("JDVP_LLM_PROVIDER", None)
                else:
                    __import__("os").environ["JDVP_LLM_PROVIDER"] = previous_provider
                if previous_model is None:
                    __import__("os").environ.pop("JDVP_LLM_MODEL", None)
                else:
                    __import__("os").environ["JDVP_LLM_MODEL"] = previous_model
                if previous_path is None:
                    __import__("os").environ.pop("JDVP_LLM_STATIC_RESPONSE_PATH", None)
                else:
                    __import__("os").environ["JDVP_LLM_STATIC_RESPONSE_PATH"] = previous_path

        output = track.extract(
            interaction_id="session-1",
            turn_number=2,
            human_input="What would you choose if you were me?",
            ai_response="I would choose option A.",
            context_turns=[],
            context_module="general",
        )
        self.assertEqual(output.jsv_hint["judgment_holder"], "AI")

    def test_static_response_provider_supports_turn_aware_payloads(self) -> None:
        provider = StaticResponseProvider(
            response_text="""
            {
              "responses_by_turn": {
                "0": {
                  "judgment_holder": "Human",
                  "delegation_awareness": "Explicit",
                  "cognitive_engagement": "Active",
                  "information_seeking": "Active",
                  "confidence": {
                    "judgment_holder": "high",
                    "delegation_awareness": "high",
                    "cognitive_engagement": "high",
                    "information_seeking": "high"
                  },
                  "evidence_spans": [
                    {"text": "help deciding", "category": "decision_support_signal"}
                  ]
                }
              },
              "default_response": {
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
                  {"text": "recommend", "category": "delegation_signal"}
                ]
              }
            }
            """,
        )
        response = provider.generate(
            system_prompt="system",
            user_prompt="interaction_id: session-1\nturn_number: 0\nhuman_input: hi\n",
        )
        normalized = normalize_llm_response(response)
        self.assertEqual(normalized["jsv_hint"]["judgment_holder"], "Human")


if __name__ == "__main__":
    unittest.main()
