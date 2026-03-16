from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.catalog.sqlite_store import CatalogRunRecord, CatalogStore


ROOT = Path(__file__).resolve().parents[1]


class ListFailedRunsTests(unittest.TestCase):
    def test_catalog_can_summarize_failed_runs_by_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                store = CatalogStore(db_path)
                store.initialize()
                with store._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO dataset_items (
                            dataset_id, interaction_id, scenario_id, blueprint_id,
                            split, relative_path, turn_count, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "generated/synthetic-general/v1",
                            "interaction-1",
                            "travel-planning",
                            "human-shared-ai",
                            "test",
                            "interactions/interaction-1.json",
                            3,
                            "2026-03-16T00:00:00Z",
                        ),
                    )
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-run-1",
                        interaction_id="interaction-1",
                        dataset_id="generated/synthetic-general/v1",
                        track_name="llm_observer",
                        model_id="test-model",
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-run-1"),
                        status="failed",
                        error_message="provider timeout",
                    )
                )
                summary = store.summarize_runs_by_scenario(status="failed")
                self.assertEqual(summary[0]["scenario_id"], "travel-planning")
                self.assertEqual(summary[0]["run_count"], 1)
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_catalog_lists_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                store = CatalogStore(db_path)
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-run-2",
                        interaction_id="interaction-2",
                        dataset_id=None,
                        track_name="llm_observer",
                        model_id="test-model",
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-run-2"),
                        status="failed",
                        error_message="provider timeout",
                    )
                )
                rows = store.list_runs(status="failed", limit=5)
                self.assertEqual(rows[0]["run_id"], "failed-run-2")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous

    def test_catalog_lists_failed_runs_for_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                store = CatalogStore(db_path)
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-run-dataset-a",
                        interaction_id="interaction-a",
                        dataset_id="generated/synthetic-general/v1",
                        track_name="llm_observer",
                        model_id="test-model",
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-run-dataset-a"),
                        status="failed",
                        error_message="provider timeout",
                    )
                )
                store.upsert_run(
                    CatalogRunRecord(
                        run_id="failed-run-dataset-b",
                        interaction_id="interaction-b",
                        dataset_id="generated/synthetic-general-rich/v1",
                        track_name="llm_observer",
                        model_id="test-model",
                        input_path=str(ROOT / "data" / "fixtures" / "sample_interaction.json"),
                        run_dir=str(Path(tmp_dir) / "runs" / "failed-run-dataset-b"),
                        status="failed",
                        error_message="provider timeout",
                    )
                )
                rows = store.list_runs(
                    status="failed",
                    dataset_id="generated/synthetic-general/v1",
                    limit=10,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["run_id"], "failed-run-dataset-a")
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous
