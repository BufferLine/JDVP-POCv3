"""Lightweight SQLite catalog for operational recovery and experiment tracking."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared_utils import utc_now


DEFAULT_CATALOG_PATH = Path("data/catalog/pocv3.sqlite3")


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
    dataset_run_id: str | None
    track_name: str
    model_id: str | None
    input_path: str
    run_dir: str
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class CatalogDatasetRunRecord:
    dataset_run_id: str
    dataset_id: str
    track_name: str
    output_root: str
    summary_path: str
    split: str | None
    scenario_id: str | None
    item_count: int
    completed_count: int
    failed_count: int
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class CatalogDatasetGenerationRunRecord:
    generation_run_id: str
    dataset_id: str
    dataset_root: str
    generation_mode: str
    scenario_pack_path: str
    target_item_count: int
    accepted_count: int
    failed_count: int
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class CatalogDatasetGenerationItemRecord:
    generation_run_id: str
    item_id: str
    interaction_id: str
    scenario_id: str
    sample_index: int
    relative_path: str
    status: str
    attempt_count: int
    item_payload_json: str | None = None
    error_message: str | None = None


class CatalogStore:
    """Lightweight SQLite catalog with a single persistent connection.

    The connection is created lazily on first use and reused for all subsequent
    operations.  Schema initialization also runs exactly once per instance
    (guarded by ``_initialized``).

    Thread-safety note: this class assumes single-threaded use (one instance
    per thread/process).  Pass ``check_same_thread=False`` only if you manage
    external locking yourself.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or resolve_catalog_path()
        self._conn: sqlite3.Connection | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return the persistent connection, creating it on first call."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Run schema migrations exactly once per instance."""
        if self._initialized:
            return
        conn = self._connect()
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
                dataset_run_id TEXT,
                track_name TEXT NOT NULL,
                model_id TEXT,
                input_path TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_runs (
                dataset_run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                track_name TEXT NOT NULL,
                output_root TEXT NOT NULL,
                summary_path TEXT NOT NULL,
                split TEXT,
                scenario_id TEXT,
                item_count INTEGER NOT NULL,
                completed_count INTEGER NOT NULL,
                failed_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_generation_runs (
                generation_run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                dataset_root TEXT NOT NULL,
                generation_mode TEXT NOT NULL,
                scenario_pack_path TEXT NOT NULL,
                target_item_count INTEGER NOT NULL,
                accepted_count INTEGER NOT NULL,
                failed_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_generation_items (
                generation_run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                interaction_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                item_payload_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (generation_run_id, item_id),
                FOREIGN KEY (generation_run_id) REFERENCES dataset_generation_runs(generation_run_id)
            );
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jdvp_runs)").fetchall()
        }
        if "dataset_run_id" not in columns:
            conn.execute("ALTER TABLE jdvp_runs ADD COLUMN dataset_run_id TEXT")
        conn.commit()
        self._initialized = True

    def _conn_ready(self) -> sqlite3.Connection:
        """Ensure initialized and return the connection."""
        self.initialize()
        return self._connect()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection.  Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    def __enter__(self) -> "CatalogStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def upsert_dataset(self, record: CatalogDatasetRecord, items: list[dict[str, Any]]) -> None:
        timestamp = utc_now()
        conn = self._conn_ready()
        with conn:
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
        conn = self._conn_ready()
        with conn:
            conn.execute(
                """
                INSERT INTO jdvp_runs (
                    run_id, interaction_id, dataset_id, dataset_run_id, track_name,
                    model_id, input_path, run_dir, status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    interaction_id=excluded.interaction_id,
                    dataset_id=excluded.dataset_id,
                    dataset_run_id=excluded.dataset_run_id,
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
                    record.dataset_run_id,
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

    def upsert_dataset_run(self, record: CatalogDatasetRunRecord) -> None:
        timestamp = utc_now()
        conn = self._conn_ready()
        with conn:
            conn.execute(
                """
                INSERT INTO dataset_runs (
                    dataset_run_id, dataset_id, track_name, output_root, summary_path,
                    split, scenario_id, item_count, completed_count, failed_count,
                    status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_run_id) DO UPDATE SET
                    dataset_id=excluded.dataset_id,
                    track_name=excluded.track_name,
                    output_root=excluded.output_root,
                    summary_path=excluded.summary_path,
                    split=excluded.split,
                    scenario_id=excluded.scenario_id,
                    item_count=excluded.item_count,
                    completed_count=excluded.completed_count,
                    failed_count=excluded.failed_count,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    record.dataset_run_id,
                    record.dataset_id,
                    record.track_name,
                    record.output_root,
                    record.summary_path,
                    record.split,
                    record.scenario_id,
                    record.item_count,
                    record.completed_count,
                    record.failed_count,
                    record.status,
                    record.error_message,
                    timestamp,
                    timestamp,
                ),
            )

    def upsert_dataset_generation_run(self, record: CatalogDatasetGenerationRunRecord) -> None:
        timestamp = utc_now()
        conn = self._conn_ready()
        with conn:
            conn.execute(
                """
                INSERT INTO dataset_generation_runs (
                    generation_run_id, dataset_id, dataset_root, generation_mode,
                    scenario_pack_path, target_item_count, accepted_count, failed_count,
                    status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_run_id) DO UPDATE SET
                    dataset_id=excluded.dataset_id,
                    dataset_root=excluded.dataset_root,
                    generation_mode=excluded.generation_mode,
                    scenario_pack_path=excluded.scenario_pack_path,
                    target_item_count=excluded.target_item_count,
                    accepted_count=excluded.accepted_count,
                    failed_count=excluded.failed_count,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    record.generation_run_id,
                    record.dataset_id,
                    record.dataset_root,
                    record.generation_mode,
                    record.scenario_pack_path,
                    record.target_item_count,
                    record.accepted_count,
                    record.failed_count,
                    record.status,
                    record.error_message,
                    timestamp,
                    timestamp,
                ),
            )

    def upsert_dataset_generation_item(self, record: CatalogDatasetGenerationItemRecord) -> None:
        timestamp = utc_now()
        conn = self._conn_ready()
        with conn:
            conn.execute(
                """
                INSERT INTO dataset_generation_items (
                    generation_run_id, item_id, interaction_id, scenario_id, sample_index,
                    relative_path, status, attempt_count, item_payload_json, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_run_id, item_id) DO UPDATE SET
                    interaction_id=excluded.interaction_id,
                    scenario_id=excluded.scenario_id,
                    sample_index=excluded.sample_index,
                    relative_path=excluded.relative_path,
                    status=excluded.status,
                    attempt_count=excluded.attempt_count,
                    item_payload_json=excluded.item_payload_json,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    record.generation_run_id,
                    record.item_id,
                    record.interaction_id,
                    record.scenario_id,
                    record.sample_index,
                    record.relative_path,
                    record.status,
                    record.attempt_count,
                    record.item_payload_json,
                    record.error_message,
                    timestamp,
                    timestamp,
                ),
            )

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        conn = self._conn_ready()
        row = conn.execute("SELECT * FROM jdvp_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def fetch_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        conn = self._conn_ready()
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return dict(row) if row is not None else None

    def fetch_dataset_run(self, dataset_run_id: str) -> dict[str, Any] | None:
        conn = self._conn_ready()
        row = conn.execute(
            "SELECT * FROM dataset_runs WHERE dataset_run_id = ?",
            (dataset_run_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def fetch_dataset_generation_run(self, generation_run_id: str) -> dict[str, Any] | None:
        conn = self._conn_ready()
        row = conn.execute(
            "SELECT * FROM dataset_generation_runs WHERE generation_run_id = ?",
            (generation_run_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_dataset_generation_items(self, *, generation_run_id: str) -> list[dict[str, Any]]:
        conn = self._conn_ready()
        rows = conn.execute(
            """
            SELECT * FROM dataset_generation_items
            WHERE generation_run_id = ?
            ORDER BY scenario_id, sample_index
            """,
            (generation_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_dataset_generation_runs(
        self,
        *,
        status: str | None = None,
        dataset_id: str | None = None,
        generation_mode: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._conn_ready()
        query = "SELECT * FROM dataset_generation_runs"
        conditions: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)
        if dataset_id is not None:
            conditions.append("dataset_id = ?")
            parameters.append(dataset_id)
        if generation_mode is not None:
            conditions.append("generation_mode = ?")
            parameters.append(generation_mode)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC, generation_run_id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]

    def list_failed_dataset_generation_items(
        self,
        *,
        generation_run_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._conn_ready()
        query = """
            SELECT * FROM dataset_generation_items
            WHERE generation_run_id = ? AND status IN ('failed', 'rejected')
            ORDER BY scenario_id, sample_index
        """
        parameters: list[Any] = [generation_run_id]
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]

    def list_runs(
        self,
        *,
        status: str | None = None,
        dataset_id: str | None = None,
        dataset_run_id: str | None = None,
        scenario_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._conn_ready()
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
        if dataset_run_id is not None:
            conditions.append("jdvp_runs.dataset_run_id = ?")
            parameters.append(dataset_run_id)
        if scenario_id is not None:
            conditions.append("dataset_items.scenario_id = ?")
            parameters.append(scenario_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY jdvp_runs.updated_at DESC, jdvp_runs.run_id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]

    def list_dataset_runs(
        self,
        *,
        status: str | None = None,
        dataset_id: str | None = None,
        scenario_id: str | None = None,
        track_name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._conn_ready()
        query = "SELECT * FROM dataset_runs"
        conditions: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)
        if dataset_id is not None:
            conditions.append("dataset_id = ?")
            parameters.append(dataset_id)
        if scenario_id is not None:
            conditions.append("scenario_id = ?")
            parameters.append(scenario_id)
        if track_name is not None:
            conditions.append("track_name = ?")
            parameters.append(track_name)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC, dataset_run_id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]

    def summarize_runs_by_scenario(self, *, status: str | None = None) -> list[dict[str, Any]]:
        conn = self._conn_ready()
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
        rows = conn.execute(query, tuple(parameters)).fetchall()
        return [dict(row) for row in rows]
