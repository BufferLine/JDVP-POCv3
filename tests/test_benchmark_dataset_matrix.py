from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_dataset_matrix import benchmark_matrix
from scripts.run_dataset_matrix import run_matrix
from src.dataset.generate_dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkDatasetMatrixTests(unittest.TestCase):
    def test_benchmark_matrix_writes_pairwise_summary(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=output_root / "generated",
                scenario_pack_path=scenario_pack,
                count_per_scenario=2,
                seed=11,
            )
            matrix_config_path = output_root / "matrix-config.json"
            matrix_config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "output_root": str(output_root / "matrix-runs"),
                        "split": "test",
                        "runs": [
                            {"label": "fixture", "track_name": "fixture_hint"},
                            {"label": "heuristic", "track_name": "heuristic_baseline"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            matrix_summary_path = run_matrix(config_path=matrix_config_path)
            benchmark_summary_path = benchmark_matrix(
                matrix_summary_path=matrix_summary_path,
                baseline_label="fixture",
                output_root=output_root / "benchmark-matrix",
            )
            summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "pocv3-benchmark-matrix-summary-v1")
            self.assertEqual(summary["baseline_label"], "fixture")
            self.assertEqual(len(summary["comparisons"]), 1)
            self.assertEqual(summary["comparisons"][0]["label"], "heuristic")
            self.assertTrue(
                (output_root / "benchmark-matrix" / "pairwise" / "heuristic").is_dir()
            )
