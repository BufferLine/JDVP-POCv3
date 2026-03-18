from __future__ import annotations

import json
import unittest
from http.client import HTTPMessage
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib import error as urllib_error

from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response
from src.method.tracks.llm_observer import LLMObserverTrack, OpenAICompatibleProvider, StaticResponseProvider, create_env_backed_llm_track


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


class SequenceProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


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

    def test_fenced_json_is_accepted(self) -> None:
        response = f"```json\n{VALID_RESPONSE.strip()}\n```"
        normalized = normalize_llm_response(response)
        self.assertEqual(normalized["jsv_hint"]["judgment_holder"], "AI")

    def test_partial_json_with_smart_quotes_is_salvaged(self) -> None:
        response = """
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
            {
              "text": "Okay, so given that, what’s your recommendation for where I should stay?”, "
        """
        normalized = normalize_llm_response(response)
        self.assertEqual(normalized["jsv_hint"]["judgment_holder"], "AI")
        self.assertEqual(normalized["evidence_spans"][0]["category"], "salvaged_evidence")


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

    def test_llm_track_retries_after_invalid_response(self) -> None:
        provider = SequenceProvider(
            [
                "I think this looks like AI delegation.",
                VALID_RESPONSE,
            ]
        )
        track = LLMObserverTrack(provider=provider, model_id="fake-model")
        output = track.extract(
            interaction_id="session-1",
            turn_number=2,
            human_input="What would you choose if you were me?",
            ai_response="I would choose option A.",
            context_turns=[],
            context_module="general",
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(output.jsv_hint["judgment_holder"], "AI")

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


_GOOD_RESPONSE_BODY = json.dumps({
    "choices": [{"message": {"content": VALID_RESPONSE.strip()}}]
}).encode("utf-8")


def _make_http_error(code: int, headers: dict[str, str] | None = None) -> urllib_error.HTTPError:
    msg = HTTPMessage()
    if headers:
        for k, v in headers.items():
            msg[k] = v
    return urllib_error.HTTPError(url="http://fake", code=code, msg="error", hdrs=msg, fp=None)


class OpenAICompatibleProviderRetryTests(unittest.TestCase):
    def _make_provider(self, max_retries: int = 3) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            base_url="http://fake",
            model="test-model",
            api_key="test-key",
            max_retries=max_retries,
        )

    def test_succeeds_on_first_attempt_no_retry(self) -> None:
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = _GOOD_RESPONSE_BODY

        with patch("src.method.tracks.llm_observer.request.urlopen", return_value=mock_response) as mock_open:
            with patch("src.method.tracks.llm_observer.time.sleep") as mock_sleep:
                result = provider._request(endpoint="http://fake/chat/completions", payload={"model": "x"})

        self.assertIn("choices", result)
        self.assertEqual(mock_open.call_count, 1)
        mock_sleep.assert_not_called()

    def test_retries_on_503_then_succeeds(self) -> None:
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = _GOOD_RESPONSE_BODY

        side_effects = [
            _make_http_error(503),
            _make_http_error(503),
            mock_response,
        ]

        with patch("src.method.tracks.llm_observer.request.urlopen", side_effect=side_effects) as mock_open:
            with patch("src.method.tracks.llm_observer.time.sleep") as mock_sleep:
                result = provider._request(endpoint="http://fake/chat/completions", payload={"model": "x"})

        self.assertIn("choices", result)
        self.assertEqual(mock_open.call_count, 3)
        # backoff: 1s before attempt 1, 2s before attempt 2
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_sleep.call_args_list[0][0][0], 1)
        self.assertEqual(mock_sleep.call_args_list[1][0][0], 2)

    def test_raises_after_max_retries_exhausted(self) -> None:
        provider = self._make_provider(max_retries=2)

        with patch("src.method.tracks.llm_observer.request.urlopen", side_effect=_make_http_error(500)):
            with patch("src.method.tracks.llm_observer.time.sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    provider._request(endpoint="http://fake/chat/completions", payload={"model": "x"})

        self.assertIn("retries", str(ctx.exception))

    def test_retries_on_429_with_retry_after_header(self) -> None:
        provider = self._make_provider(max_retries=1)
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = _GOOD_RESPONSE_BODY

        side_effects = [
            _make_http_error(429, {"Retry-After": "5"}),
            mock_response,
        ]

        sleep_calls: list[float] = []

        def capture_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with patch("src.method.tracks.llm_observer.request.urlopen", side_effect=side_effects):
            with patch("src.method.tracks.llm_observer.time.sleep", side_effect=capture_sleep):
                result = provider._request(endpoint="http://fake/chat/completions", payload={"model": "x"})

        self.assertIn("choices", result)
        # First sleep is the exponential backoff (1s for attempt 1),
        # second sleep is the Retry-After remainder (5 - 0 = 5s on attempt 0)
        self.assertGreaterEqual(len(sleep_calls), 1)

    def test_retries_on_connection_error(self) -> None:
        provider = self._make_provider(max_retries=2)
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = _GOOD_RESPONSE_BODY

        side_effects = [ConnectionError("connection refused"), mock_response]

        with patch("src.method.tracks.llm_observer.request.urlopen", side_effect=side_effects) as mock_open:
            with patch("src.method.tracks.llm_observer.time.sleep"):
                result = provider._request(endpoint="http://fake/chat/completions", payload={"model": "x"})

        self.assertIn("choices", result)
        self.assertEqual(mock_open.call_count, 2)

    def test_json_mode_fallback_not_retried(self) -> None:
        """400 with response_format present triggers the json_mode fallback, not the retry loop."""
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = _GOOD_RESPONSE_BODY

        side_effects = [
            _make_http_error(400),  # triggers json_mode fallback
            mock_response,          # fallback succeeds
        ]

        with patch("src.method.tracks.llm_observer.request.urlopen", side_effect=side_effects) as mock_open:
            with patch("src.method.tracks.llm_observer.time.sleep") as mock_sleep:
                result = provider._request(
                    endpoint="http://fake/chat/completions",
                    payload={"model": "x", "response_format": {"type": "json_object"}},
                )

        self.assertIn("choices", result)
        self.assertEqual(mock_open.call_count, 2)
        mock_sleep.assert_not_called()

    def test_non_retryable_http_error_raises_immediately(self) -> None:
        provider = self._make_provider()

        with patch("src.method.tracks.llm_observer.request.urlopen", side_effect=_make_http_error(401)) as mock_open:
            with patch("src.method.tracks.llm_observer.time.sleep") as mock_sleep:
                with self.assertRaises(RuntimeError):
                    provider._request(endpoint="http://fake/chat/completions", payload={"model": "x"})

        self.assertEqual(mock_open.call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
