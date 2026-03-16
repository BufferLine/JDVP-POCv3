from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_fewshot_regression_suite import run_suite


ROOT = Path(__file__).resolve().parents[1]


class RegressionSuiteTests(unittest.TestCase):
    def test_run_suite_writes_manifest_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "suite-output"
            manifest_path = run_suite(
                config_path=ROOT / "config" / "eval" / "fewshot_regression_general_v1.json",
                output_root=output_root,
                clean=True,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            results_path = Path(manifest["results_path"])
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["item_count"], results["item_count"])
            self.assertEqual(results["comparison_track"], "heuristic_baseline")
            self.assertIn("scenario_breakdown", results)


if __name__ == "__main__":
    unittest.main()
