from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_validation_suite import run_validation_suite


class ValidationSuiteTests(unittest.TestCase):
    def test_validation_suite_writes_summary_and_run_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "validation"
            summary_path = run_validation_suite(
                output_root=output_root,
                include_pytest=False,
                include_regression=False,
                require_upstream=False,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "pocv3-validation-suite-v1")
            self.assertEqual(summary["steps"]["contracts"]["status"], "passed")
            self.assertEqual(summary["steps"]["pytest"]["status"], "skipped")
            self.assertEqual(summary["steps"]["fewshot_regression"]["status"], "skipped")

            smoke_run_dir = Path(summary["steps"]["smoke_run"]["run_dir"])
            baseline_run_dir = Path(summary["steps"]["baseline_run"]["run_dir"])
            self.assertTrue((smoke_run_dir / "manifest.json").is_file())
            self.assertTrue((baseline_run_dir / "overlays" / "track_outputs.jsonl").is_file())

    def test_validation_suite_surfaces_regression_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "validation"
            summary_path = run_validation_suite(
                output_root=output_root,
                include_pytest=False,
                include_regression=True,
                require_upstream=False,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            regression = summary["steps"]["fewshot_regression"]
            self.assertEqual(regression["status"], "passed")
            self.assertEqual(regression["summary"]["item_count"], 2)
            self.assertGreater(len(regression["summary"]["top_disagreement_fields"]), 0)
            self.assertLessEqual(regression["summary"]["average_field_disagreement_rate"], 0.42)
            self.assertLessEqual(regression["summary"]["max_field_disagreement_rate"], 0.42)
            self.assertGreaterEqual(regression["summary"]["zero_disagreement_items"], 0)


if __name__ == "__main__":
    unittest.main()
