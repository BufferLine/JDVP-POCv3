from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.catalog.sqlite_store import CatalogStore
from src.dataset.generate_dataset import generate_dataset
from src.service.errors import ServiceError
from src.service.poc_service import RunRequest, run_interaction


ROOT = Path(__file__).resolve().parents[1]


class CatalogStoreTests(unittest.TestCase):
    def test_generate_dataset_registers_catalog_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general",
                    dataset_version="v1",
                    output_root=Path(tmp_dir) / "generated",
                    scenario_pack_path=ROOT / "config" / "datasets" / "general_scenarios_v1.json",
                    count_per_scenario=1,
                    seed=11,
                )
                store = CatalogStore(db_path)
                dataset = store.fetch_dataset("generated/synthetic-general/v1")
                self.assertIsNotNone(dataset)
                self.assertEqual(dataset["dataset_root"], str(dataset_root))
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_run_interaction_registers_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                with self.assertRaises(ServiceError):
                    run_interaction(
                        RunRequest(
                            input_path=Path(tmp_dir) / "missing.json",
                            run_id="failed-run",
                            output_root=Path(tmp_dir) / "runs",
                            track_name="fixture_hint",
                        )
                    )
                store = CatalogStore(db_path)
                run = store.fetch_run("failed-run")
                self.assertIsNotNone(run)
                self.assertEqual(run["status"], "failed")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_run_dataset_registers_dataset_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general",
                    dataset_version="v1",
                    output_root=Path(tmp_dir) / "generated",
                    scenario_pack_path=ROOT / "config" / "datasets" / "general_scenarios_v1.json",
                    count_per_scenario=1,
                    seed=11,
                )
                from src.service.dataset_run_service import DatasetRunRequest, run_dataset

                result = run_dataset(
                    DatasetRunRequest(
                        dataset_root=dataset_root,
                        output_root=Path(tmp_dir) / "dataset-runs" / "fixture",
                        track_name="fixture_hint",
                        split="test",
                    )
                )
                store = CatalogStore(db_path)
                dataset_run = store.fetch_dataset_run(result.dataset_run_id)
                self.assertIsNotNone(dataset_run)
                self.assertEqual(dataset_run["status"], "completed")
                self.assertEqual(dataset_run["dataset_id"], "generated/synthetic-general/v1")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_generate_dataset_registers_generation_run_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general",
                    dataset_version="v1",
                    output_root=Path(tmp_dir) / "generated",
                    scenario_pack_path=ROOT / "config" / "datasets" / "general_scenarios_v1.json",
                    count_per_scenario=1,
                    seed=11,
                )
                store = CatalogStore(db_path)
                generation_run_id = f"{dataset_root.resolve(strict=False)}::template::seed=11::count=1"
                generation_run = store.fetch_dataset_generation_run(generation_run_id)
                self.assertIsNotNone(generation_run)
                self.assertEqual(generation_run["status"], "completed")
                generation_items = store.list_dataset_generation_items(generation_run_id=generation_run_id)
                self.assertEqual(len(generation_items), 3)
                self.assertTrue(all(row["status"] == "accepted" for row in generation_items))
                listed_runs = store.list_dataset_generation_runs(status="completed", limit=10)
                self.assertEqual(len(listed_runs), 1)
                self.assertEqual(listed_runs[0]["generation_run_id"], generation_run_id)
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
