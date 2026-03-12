from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataset.generate_dataset import generate_dataset
from src.dataset.build_fewshot_pack import build_fewshot_pack
from src.eval.fewshot_benchmark import build_fewshot_benchmark_plan
from src.method.fewshot.selector import load_fewshot_pack, select_examples
from src.pipeline.run_poc import run_poc


ROOT = Path(__file__).resolve().parents[1]


class FewshotBenchmarkTests(unittest.TestCase):
    def test_selector_excludes_target_interaction(self) -> None:
        pack = {
            "examples": [
                {"interaction_id": "a", "context_module": "general", "jsv_hint": {"judgment_holder": "Human"}},
                {"interaction_id": "b", "context_module": "general", "jsv_hint": {"judgment_holder": "Shared"}},
                {"interaction_id": "c", "context_module": "general", "jsv_hint": {"judgment_holder": "AI"}},
                {"interaction_id": "d", "context_module": "financial", "jsv_hint": {"judgment_holder": "AI"}},
            ]
        }
        selected = select_examples(
            pack=pack,
            target_interaction_id="a",
            context_module="general",
            max_examples=2,
        )
        self.assertEqual([example["interaction_id"] for example in selected], ["c", "b"])

    def test_benchmark_plan_uses_dataset_split(self) -> None:
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
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root / "runs",
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=3)
            plan_path = output_root / "fewshot-test-plan.json"
            build_fewshot_benchmark_plan(
                dataset_root=dataset_root,
                fewshot_pack_path=pack_path,
                split="test",
                output_path=plan_path,
                max_examples=2,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["split"], "test")
            self.assertEqual(plan["item_count"], 2)
            self.assertEqual(len(plan["items"]), 2)
            self.assertTrue(plan["items"][0]["input_path"].endswith(".json"))
            self.assertLessEqual(plan["items"][0]["fewshot_example_count"], 2)


if __name__ == "__main__":
    unittest.main()
