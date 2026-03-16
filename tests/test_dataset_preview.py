from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.preview_dataset import build_preview
from src.dataset.generate_dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]


class DatasetPreviewTests(unittest.TestCase):
    def test_build_preview_includes_turn_text_and_blueprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general-rich",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=ROOT / "config" / "datasets" / "general_scenarios_v2.json",
                count_per_scenario=2,
                seed=11,
            )
            preview = build_preview(dataset_root=dataset_root, max_items=3)
            self.assertEqual(preview["schema_version"], "pocv3-dataset-preview-v1")
            self.assertLessEqual(preview["item_count"], 3)
            self.assertIn("blueprint_id", preview["items"][0])
            self.assertIn("turns", preview["items"][0])
            self.assertIn("human_input", preview["items"][0]["turns"][0])
            self.assertIn("jsv_hint", preview["items"][0]["turns"][0])

    def test_preview_script_writes_preview_file_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general-rich",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=ROOT / "config" / "datasets" / "general_scenarios_v2.json",
                count_per_scenario=1,
                seed=5,
            )
            preview = build_preview(dataset_root=dataset_root, max_items=2)
            preview_path = Path(tmp_dir) / "preview.json"
            preview_path.write_text(json.dumps(preview), encoding="utf-8")
            loaded = json.loads(preview_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["item_count"], len(loaded["items"]))


if __name__ == "__main__":
    unittest.main()
