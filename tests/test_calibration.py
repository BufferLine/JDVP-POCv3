from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.calibration import build_calibration_report, load_calibration_dataset


class CalibrationTests(unittest.TestCase):
    def test_reports_human_agreement_and_adjudicated_observer_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": "jdvp-human-calibration-v1",
                "dataset_id": "human-ai-pilot-v1",
                "measurement_profile": "human_ai",
                "consent_basis": "documented-research-consent",
            }), encoding="utf-8")
            row = {
                "interaction_id": "i-1",
                "turn_number": 0,
                "annotations": [
                    {"annotator_id": "a-1", "labels": {
                        "judgment_holder": 2, "delegation_awareness": 2,
                        "cognitive_engagement": 2, "information_seeking": 2,
                    }},
                    {"annotator_id": "a-2", "labels": {
                        "judgment_holder": 3, "delegation_awareness": 2,
                        "cognitive_engagement": 2, "information_seeking": 2,
                    }},
                ],
                "adjudicated_labels": {
                    "judgment_holder": 2, "delegation_awareness": 2,
                    "cognitive_engagement": 2, "information_seeking": 2,
                },
            }
            (root / "annotations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest, rows = load_calibration_dataset(root)

        report = build_calibration_report(
            manifest=manifest,
            rows=rows,
            predictions={
                ("i-1", 0): {
                    "judgment_holder": 2, "delegation_awareness": 3,
                    "cognitive_engagement": 2, "information_seeking": 2,
                },
            },
            run_metadata=[{"latency_ms": 4.0, "estimated_cost_usd": 0.0}],
        )

        self.assertEqual(report["adjudicated_rows_evaluated"], 1)
        self.assertEqual(report["observer_performance"]["judgment_holder"]["exact_agreement"], 1.0)
        self.assertEqual(report["observer_performance"]["delegation_awareness"]["mean_absolute_error"], 1.0)
        self.assertEqual(report["annotator_agreement"]["judgment_holder"]["pairwise_exact_agreement"], 0.0)
        self.assertEqual(report["run_cost_and_latency"]["mean_latency_ms"], 4.0)

    def test_rejects_dataset_without_consent_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": "jdvp-human-calibration-v1",
                "measurement_profile": "human_ai",
            }), encoding="utf-8")
            (root / "annotations.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration_dataset(root)

    def test_latency_p95_uses_nearest_rank_for_small_runs(self) -> None:
        report = build_calibration_report(
            manifest={"measurement_profile": "human_ai"},
            rows=[],
            predictions={},
            run_metadata=[{"latency_ms": 1.0}, {"latency_ms": 2.0}, {"latency_ms": 100.0}],
        )

        self.assertEqual(report["run_cost_and_latency"]["p95_latency_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
