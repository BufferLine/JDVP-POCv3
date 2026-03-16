from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_dataset_matrix import run_matrix
from src.dataset.generate_dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]


class DatasetMatrixTests(unittest.TestCase):
    def test_run_matrix_writes_summary_for_multiple_tracks(self) -> None:
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
            config_path = output_root / "matrix-config.json"
            config_path.write_text(
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
            summary_path = run_matrix(config_path=config_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "pocv3-dataset-matrix-summary-v1")
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(summary["runs"][0]["completed_count"], 2)
            self.assertEqual(summary["runs"][1]["failed_count"], 0)
