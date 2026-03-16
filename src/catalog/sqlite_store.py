"""Lightweight SQLite catalog for operational recovery and experiment tracking."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_PATH = Path("data/catalog/pocv3.sqlite3")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_catalog_path() -> Path:
    configured = os.getenv("JDVP_CATALOG_DB_PATH")
    return Path(configured) if configured else DEFAULT_CATALOG_PATH


@dataclass(frozen=True)
class CatalogDatasetRecord:
    dataset_id: str
    dataset_root: str
    dataset_kind: str
    scenario_pack_id: str
    generation_seed: int
    count_per_scenario: int


@dataclass(frozen=True)
class CatalogRunRecord:
    run_id: str
    interaction_id: str | None
    dataset_id: str | None
    track_name: str
    model_id: str | None
    input_path: str
    run_dir: str
    status: str
    error_message: str | None = None


class CatalogStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or resolve_catalog_path()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    dataset_root TEXT NOT NULL,
                    dataset_kind TEXT NOT NULL,
                    scenario_pack_id TEXT NOT NULL,
                    generation_seed INTEGER NOT NULL,
                    count_per_scenario INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_items (
                    dataset_id TEXT NOT NULL,
                    interaction_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    blueprint_id TEXT,
                    split TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, interaction_id),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                );

                CREATE TABLE IF NOT EXISTS jdvp_runs (
                    run_id TEXT PRIMARY KEY,
                    interaction_id TEXT,
                    dataset_id TEXT,
                    track_name TEXT NOT NULL,
                    model_id TEXT,
                    input_path TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def upsert_dataset(self, record: CatalogDatasetRecord, items: list[dict[str, Any]]) -> None:
        timestamp = utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, dataset_root, dataset_kind, scenario_pack_id,
                    generation_seed, count_per_scenario, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    dataset_root=excluded.dataset_root,
                    dataset_kind=excluded.dataset_kind,
                    scenario_pack_id=excluded.scenario_pack_id,
                    generation_seed=excluded.generation_seed,
                    count_per_scenario=excluded.count_per_scenario,
                    updated_at=excluded.updated_at
                """,
                (
                    record.dataset_id,
                    record.dataset_root,
                    record.dataset_kind,
                    record.scenario_pack_id,
                    record.generation_seed,
                    record.count_per_scenario,
                    timestamp,
                    timestamp,
                ),
            )
            conn.execute("DELETE FROM dataset_items WHERE dataset_id = ?", (record.dataset_id,))
            conn.executemany(
                """
                INSERT INTO dataset_items (
                    dataset_id, interaction_id, scenario_id, blueprint_id,
                    split, relative_path, turn_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.dataset_id,
                        str(item["interaction_id"]),
                        str(item["scenario_id"]),
                        (
                            str(item["blueprint_id"])
                            if item.get("blueprint_id") is not None
                            else None
                        ),
                        str(item["split"]),
                        str(item["relative_path"]),
                        int(item["turn_count"]),
                        timestamp,
                    )
                    for item in items
                ],
            )

    def upsert_run(self, record: CatalogRunRecord) -> None:
        timestamp = utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jdvp_runs (
                    run_id, interaction_id, dataset_id, track_name, model_id,
                    input_path, run_dir, status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    interaction_id=excluded.interaction_id,
                    dataset_id=excluded.dataset_id,
                    track_name=excluded.track_name,
                    model_id=excluded.model_id,
                    input_path=excluded.input_path,
                    run_dir=excluded.run_dir,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    record.run_id,
                    record.interaction_id,
                    record.dataset_id,
                    record.track_name,
                    record.model_id,
                    record.input_path,
                    record.run_dir,
                    record.status,
                    record.error_message,
                    timestamp,
                    timestamp,
                ),
            )

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jdvp_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def fetch_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_runs(
        self,
        *,
        status: str | None = None,
        dataset_id: str | None = None,
        scenario_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize()
        query = """
            SELECT
                jdvp_runs.*,
                dataset_items.scenario_id AS catalog_scenario_id,
                dataset_items.blueprint_id AS catalog_blueprint_id
            FROM jdvp_runs
            LEFT JOIN dataset_items
              ON dataset_items.interaction_id = jdvp_runs.interaction_id
        """
        conditions: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            conditions.append("jdvp_runs.status = ?")
            parameters.append(status)
        if dataset_id is not None:
            conditions.append("jdvp_runs.dataset_id = ?")
            parameters.append(dataset_id)
        if scenario_id is not None:
            conditions.append("dataset_items.scenario_id = ?")
            parameters.append(scenario_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY jdvp_runs.updated_at DESC, jdvp_runs.run_id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]

    def summarize_runs_by_scenario(self, *, status: str | None = None) -> list[dict[str, Any]]:
        self.initialize()
        query = """
            SELECT
                COALESCE(dataset_items.scenario_id, 'unknown') AS scenario_id,
                jdvp_runs.status AS status,
                COUNT(*) AS run_count
            FROM jdvp_runs
            LEFT JOIN dataset_items
              ON dataset_items.interaction_id = jdvp_runs.interaction_id
        """
        parameters: list[Any] = []
        if status is not None:
            query += " WHERE jdvp_runs.status = ?"
            parameters.append(status)
        query += """
            GROUP BY COALESCE(dataset_items.scenario_id, 'unknown'), jdvp_runs.status
            ORDER BY run_count DESC, scenario_id
        """
        with self._connect() as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]
