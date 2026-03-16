from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.rerun_failed_runs import rerun_failed_runs
from src.catalog.sqlite_store import CatalogRunRecord, CatalogStore


ROOT = Path(__file__).resolve().parents[1]


class RerunFailedRunsTests(unittest.TestCase):
    def test_rerun_failed_runs_replays_selected_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                store = CatalogStore(db_path)
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-fixture-run",
                        interaction_id="fixture-general-001",
                        dataset_id=None,
                        dataset_run_id=None,
                        track_name="fixture_hint",
                        model_id=None,
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-fixture-run"),
                        status="failed",
                        error_message="provider request failed",
                    )
                )
                results = rerun_failed_runs(catalog=store, limit=1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["status"], "completed")
                rerun_record = store.fetch_run(results[0]["rerun_run_id"])
                self.assertIsNotNone(rerun_record)
                self.assertEqual(rerun_record["status"], "completed")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_rerun_failed_runs_filters_by_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                store = CatalogStore(db_path)
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-dataset-run",
                        interaction_id="fixture-general-001",
                        dataset_id="generated/synthetic-general/v1",
                        dataset_run_id="dataset-run-a",
                        track_name="fixture_hint",
                        model_id=None,
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-dataset-run"),
                        status="failed",
                        error_message="provider request failed",
                    )
                )
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-other-run",
                        interaction_id="fixture-general-001",
                        dataset_id="generated/other/v1",
                        dataset_run_id="dataset-run-b",
                        track_name="fixture_hint",
                        model_id=None,
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-other-run"),
                        status="failed",
                        error_message="provider request failed",
                    )
                )
                results = rerun_failed_runs(
                    catalog=store,
                    dataset_id="generated/synthetic-general/v1",
                    limit=10,
                )
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["original_run_id"], "failed-dataset-run")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_rerun_failed_runs_filters_by_dataset_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                store = CatalogStore(db_path)
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-dataset-run-a",
                        interaction_id="fixture-general-001",
                        dataset_id="generated/synthetic-general/v1",
                        dataset_run_id="dataset-run-a",
                        track_name="fixture_hint",
                        model_id=None,
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-dataset-run-a"),
                        status="failed",
                        error_message="provider request failed",
                    )
                )
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-dataset-run-b",
                        interaction_id="fixture-general-001",
                        dataset_id="generated/synthetic-general/v1",
                        dataset_run_id="dataset-run-b",
                        track_name="fixture_hint",
                        model_id=None,
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-dataset-run-b"),
                        status="failed",
                        error_message="provider request failed",
                    )
                )
                results = rerun_failed_runs(
                    catalog=store,
                    dataset_run_id="dataset-run-a",
                    limit=10,
                )
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["original_run_id"], "failed-dataset-run-a")
                rerun_record = store.fetch_run(results[0]["rerun_run_id"])
                self.assertEqual(rerun_record["dataset_run_id"], "dataset-run-a")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous
