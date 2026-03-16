from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataset.generate_dataset import generate_dataset
from src.pipeline.run_poc import run_poc


ROOT = Path(__file__).resolve().parents[1]


class DatasetGenerationTests(unittest.TestCase):
    def test_dataset_generation_writes_manifest_and_splits(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=scenario_pack,
                count_per_scenario=2,
                seed=11,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            splits = json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["dataset_id"], "generated/synthetic-general/v1")
            self.assertEqual(manifest["scenario_pack_id"], "general-scenarios-v1")
            self.assertEqual(manifest["item_count"], 6)
            self.assertEqual(len(manifest["items"]), 6)
            self.assertEqual(sum(len(values) for values in splits.values()), 6)
            self.assertTrue((dataset_root / "interactions").is_dir())
            self.assertIn("turn_variant_choices", manifest["items"][0])
            self.assertEqual(len(manifest["items"][0]["turn_variant_choices"]), 3)

    def test_generated_dataset_can_feed_pipeline(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=output_root,
                scenario_pack_path=scenario_pack,
                count_per_scenario=1,
                seed=5,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction_path = dataset_root / first_item["relative_path"]
            run_dir = run_poc(
                input_path=interaction_path,
                run_id="generated-smoke",
                output_root=output_root / "runs",
                track_name="fixture_hint",
            )
            self.assertTrue((run_dir / "canonical" / "trajectory.json").is_file())
            trajectory = json.loads((run_dir / "canonical" / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory["interaction_id"], first_item["interaction_id"])

    def test_dataset_generation_uses_turn_variants(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=scenario_pack,
                count_per_scenario=3,
                seed=11,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            travel_items = [
                item for item in manifest["items"]
                if item["scenario_id"] == "travel-planning"
            ]
            rendered_inputs = {
                json.loads((dataset_root / item["relative_path"]).read_text(encoding="utf-8"))["turns"][0]["human_input"]
                for item in travel_items
            }
            self.assertGreater(len(rendered_inputs), 1)

    def test_dataset_generation_supports_blueprint_selection(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v2.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general-rich",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=scenario_pack,
                count_per_scenario=4,
                seed=11,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scenario_pack_id"], "general-scenarios-v2")
            self.assertTrue(all("blueprint_id" in item for item in manifest["items"]))
            blueprint_ids = {item["blueprint_id"] for item in manifest["items"]}
            self.assertGreater(len(blueprint_ids), 1)
            turn_counts = {item["turn_count"] for item in manifest["items"]}
            self.assertIn(4, turn_counts)


if __name__ == "__main__":
    unittest.main()
