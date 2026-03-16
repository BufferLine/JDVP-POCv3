from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset.build_fewshot_pack import build_fewshot_pack
from src.dataset.generate_dataset import generate_dataset
from src.eval.fewshot_benchmark import build_fewshot_benchmark_plan
from src.service.json_api import handle_json_payload
from src.service.poc_service import run_interaction_file


ROOT = Path(__file__).resolve().parents[1]
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
    {"text": "Just recommend one plan for me.", "category": "delegation_signal"}
  ],
  "observer_confidence": 0.77,
  "observer_notes": "Detected direct delegation request"
}
"""


class ServiceJsonApiTests(unittest.TestCase):
    def test_handle_json_payload_runs_interaction(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_json_payload(
                {
                    "operation": "run_interaction",
                    "input_path": str(fixture),
                    "run_id": "json-api-run",
                    "output_root": tmp_dir,
                    "track_name": "fixture_hint",
                }
            )
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["run_id"], "json-api-run")

    def test_handle_json_payload_runs_fewshot_benchmark(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=output_root,
                scenario_pack_path=scenario_pack,
                count_per_scenario=2,
                seed=11,
            )
            seed_run_dir = run_interaction_file(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root / "seed-runs",
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=seed_run_dir, output_path=pack_path, max_examples=3)
            plan_path = output_root / "fewshot-test-plan.json"
            build_fewshot_benchmark_plan(
                dataset_root=dataset_root,
                fewshot_pack_path=pack_path,
                split="test",
                output_path=plan_path,
                max_examples=2,
            )
            with patch.dict(
                os.environ,
                {
                    "JDVP_LLM_BASE_URL": "http://localhost:11434/v1",
                    "JDVP_LLM_API_KEY": "dummy",
                    "JDVP_LLM_MODEL": "fake-model",
                },
                clear=False,
            ):
                with patch(
                    "src.method.tracks.llm_observer.OpenAICompatibleProvider.generate",
                    return_value=VALID_RESPONSE,
                ):
                    response = handle_json_payload(
                        {
                            "operation": "run_fewshot_benchmark",
                            "plan_path": str(plan_path),
                            "output_root": str(output_root / "benchmark-results"),
                        }
                    )
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["split"], "test")

    def test_handle_json_payload_rejects_unknown_operation(self) -> None:
        response = handle_json_payload({"operation": "unknown"})
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "unsupported_operation")


if __name__ == "__main__":
    unittest.main()
