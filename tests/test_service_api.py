from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.service import run_interaction_file
from src.service.poc_service import build_pipeline_artifacts


ROOT = Path(__file__).resolve().parents[1]


class ServiceApiTests(unittest.TestCase):
    def test_build_pipeline_artifacts_returns_canonical_sequences(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        raw_interaction = json.loads(fixture.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = build_pipeline_artifacts(
                raw_interaction=raw_interaction,
                track_name="fixture_hint",
                run_dir=Path(tmp_dir) / "service-run",
            )
            self.assertEqual(len(artifacts.jsv_sequence), 3)
            self.assertEqual(len(artifacts.dv_sequence), 2)
            self.assertEqual(artifacts.trajectory["interaction_id"], "fixture-general-001")

    def test_run_interaction_file_writes_outputs(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_interaction_file(
                input_path=fixture,
                run_id="service-run",
                output_root=Path(tmp_dir),
                track_name="fixture_hint",
            )
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "canonical" / "trajectory.json").is_file())


if __name__ == "__main__":
    unittest.main()
