from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.run_poc import run_poc


ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    def test_smoke_pipeline_writes_canonical_outputs(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_poc(
                input_path=fixture,
                run_id="smoke-test",
                output_root=Path(tmp_dir),
            )
            manifest_path = run_dir / "manifest.json"
            trajectory_path = run_dir / "canonical" / "trajectory.json"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(trajectory_path.is_file())

            with trajectory_path.open("r", encoding="utf-8") as handle:
                trajectory = json.load(handle)
            self.assertEqual(trajectory["interaction_id"], "fixture-general-001")
            self.assertEqual(len(trajectory["vectors"]), 2)


if __name__ == "__main__":
    unittest.main()
