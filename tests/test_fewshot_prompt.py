from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataset.build_fewshot_pack import build_fewshot_pack
from src.method.tracks.fewshot_prompt import FewshotPromptTrack
from src.pipeline.run_poc import run_poc


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


ROOT = Path(__file__).resolve().parents[1]


class CapturingProvider:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response_text


class FewshotPromptTests(unittest.TestCase):
    def test_build_fewshot_pack_from_run_extracts(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root,
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=2)
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            self.assertEqual(pack["schema_version"], "pocv3-fewshot-pack-v1")
            self.assertEqual(len(pack["examples"]), 2)
            self.assertEqual(pack["examples"][0]["turn_number"], 0)

    def test_fewshot_track_includes_examples_in_prompt(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root,
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=2)
            provider = CapturingProvider(VALID_RESPONSE)
            track = FewshotPromptTrack(
                provider=provider,
                model_id="fake-model",
                fewshot_pack_path=pack_path,
            )
            output = track.extract(
                interaction_id="session-1",
                turn_number=2,
                human_input="What would you choose if you were me?",
                ai_response="I would choose option A.",
                context_turns=[],
                context_module="general",
            )
            prompt = provider.calls[0]["user_prompt"]
            self.assertIn("fewshot_examples:", prompt)
            self.assertIn("\"turn_number\": 0", prompt)
            self.assertEqual(output.track_id, "fewshot_prompt")
            self.assertEqual(output.jsv_hint["judgment_holder"], "AI")


if __name__ == "__main__":
    unittest.main()
