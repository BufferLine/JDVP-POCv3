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
from src.eval.run_fewshot_benchmark import run_fewshot_benchmark_plan
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
    {"text": "Just recommend one plan for me.", "category": "delegation_signal"}
  ],
  "observer_confidence": 0.77,
  "observer_notes": "Detected direct delegation request"
}
"""


ROOT = Path(__file__).resolve().parents[1]


class RunFewshotBenchmarkTests(unittest.TestCase):
    def test_run_fewshot_benchmark_plan_writes_results(self) -> None:
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
            fixture_run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root / "seed-runs",
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=fixture_run_dir, output_path=pack_path, max_examples=3)
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
                    results_path = run_fewshot_benchmark_plan(
                        plan_path=plan_path,
                        output_root=output_root / "benchmark-results",
                    )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results["item_count"], 2)
            self.assertEqual(results["comparison_track"], "heuristic_baseline")
            self.assertEqual(len(results["items"]), 2)
            self.assertTrue(Path(results["items"][0]["fewshot_run_dir"]).is_dir())
            self.assertTrue(Path(results["items"][0]["comparison_report_dir"]).is_dir())


if __name__ == "__main__":
    unittest.main()
