from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.ensemble_benchmark import compare_runs
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

    def test_heuristic_baseline_writes_overlay_outputs(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_poc(
                input_path=fixture,
                run_id="heuristic-test",
                output_root=Path(tmp_dir),
                track_name="heuristic_baseline",
            )
            overlay_path = run_dir / "overlays" / "track_outputs.jsonl"
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(overlay_path.is_file())
            self.assertTrue(manifest_path.is_file())

            rows = overlay_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)
            first = json.loads(rows[0])
            self.assertEqual(first["track"]["track_id"], "heuristic_baseline")

    def test_run_storage_writes_extracts_and_checkpoint(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_poc(
                input_path=fixture,
                run_id="storage-test",
                output_root=Path(tmp_dir),
                track_name="heuristic_baseline",
            )
            extracts_jsonl = run_dir / "extracts" / "heuristic_baseline" / "extracts.jsonl"
            checkpoint_path = run_dir / "checkpoints" / "progress.json"
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(extracts_jsonl.is_file())
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(manifest_path.is_file())

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["status"], "completed")
            self.assertEqual(checkpoint["completed_turns"], 3)

    def test_resume_reuses_existing_turn_extracts(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="resume-test",
                output_root=output_root,
                track_name="heuristic_baseline",
            )
            manifest_before = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_before["resumed_turns"], [])

            run_dir = run_poc(
                input_path=fixture,
                run_id="resume-test",
                output_root=output_root,
                track_name="heuristic_baseline",
                resume=True,
            )
            manifest_after = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_after["resumed_turns"], [0, 1, 2])
            self.assertEqual(manifest_after["written_turns"], [])

    def test_compare_runs_writes_benchmark_outputs(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            fixture_run = run_poc(
                input_path=fixture,
                run_id="fixture-run",
                output_root=output_root,
                track_name="fixture_hint",
            )
            baseline_run = run_poc(
                input_path=fixture,
                run_id="baseline-run",
                output_root=output_root,
                track_name="heuristic_baseline",
            )
            benchmark_dir = output_root / "benchmark-report"
            summary = compare_runs(
                run_dirs=[fixture_run, baseline_run],
                output_dir=benchmark_dir,
            )
            self.assertEqual(summary["turns_compared"], 3)
            self.assertTrue((benchmark_dir / "benchmark_summary.json").is_file())
            comparisons_path = benchmark_dir / "turn_comparisons.jsonl"
            self.assertTrue(comparisons_path.is_file())
            self.assertIn("heuristic_baseline", summary["track_names"])
            rows = [json.loads(line) for line in comparisons_path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("track_field_values", rows[0])
            tfv_keys = list(rows[0]["track_field_values"].keys())
            self.assertTrue(
                any("fixture_hint" in k for k in tfv_keys),
                f"expected a key containing 'fixture_hint' in {tfv_keys}",
            )


if __name__ == "__main__":
    unittest.main()
