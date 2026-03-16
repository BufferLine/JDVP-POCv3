"""Helpers for checking and refreshing vendored JDVP schema snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.protocol_core.schema_validate import SIBLING_PROTOCOL_ROOT, VENDORED_PROTOCOL_ROOT


SCHEMA_FILENAMES = (
    "jsv-schema.json",
    "dv-schema.json",
    "trajectory-schema.json",
)
SNAPSHOT_MANIFEST_FILENAME = "schema_snapshot.json"


@dataclass(frozen=True)
class SchemaDiff:
    filename: str
    upstream_sha256: str
    vendored_sha256: str


@dataclass(frozen=True)
class SchemaSnapshotManifest:
    upstream_root: str
    upstream_revision: str | None
    files: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "pocv3-vendored-schema-snapshot-v1",
            "upstream_root": self.upstream_root,
            "upstream_revision": self.upstream_revision,
            "files": self.files,
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_schema_paths(root: Path) -> dict[str, Path]:
    return {filename: root / filename for filename in SCHEMA_FILENAMES}


def ensure_schema_root(root: Path) -> None:
    missing = [path for path in required_schema_paths(root).values() if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing schema files under {root}: {missing_text}")


def compare_schema_roots(
    upstream_root: Path = SIBLING_PROTOCOL_ROOT,
    vendored_root: Path = VENDORED_PROTOCOL_ROOT,
) -> list[SchemaDiff]:
    ensure_schema_root(upstream_root)
    ensure_schema_root(vendored_root)

    diffs: list[SchemaDiff] = []
    vendored_paths = required_schema_paths(vendored_root)
    for filename, upstream_path in required_schema_paths(upstream_root).items():
        vendored_path = vendored_paths[filename]
        upstream_sha = sha256_file(upstream_path)
        vendored_sha = sha256_file(vendored_path)
        if upstream_sha != vendored_sha:
            diffs.append(
                SchemaDiff(
                    filename=filename,
                    upstream_sha256=upstream_sha,
                    vendored_sha256=vendored_sha,
                )
            )
    return diffs


def manifest_path_for(vendored_root: Path) -> Path:
    return vendored_root.parent / SNAPSHOT_MANIFEST_FILENAME


def detect_git_revision(upstream_root: Path) -> str | None:
    repo_root = upstream_root.parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_snapshot_manifest(
    upstream_root: Path = SIBLING_PROTOCOL_ROOT,
    vendored_root: Path = VENDORED_PROTOCOL_ROOT,
    upstream_revision: str | None = None,
) -> SchemaSnapshotManifest:
    ensure_schema_root(vendored_root)
    revision = upstream_revision if upstream_revision is not None else detect_git_revision(upstream_root)
    return SchemaSnapshotManifest(
        upstream_root=str(upstream_root),
        upstream_revision=revision,
        files={
            filename: sha256_file(path)
            for filename, path in required_schema_paths(vendored_root).items()
        },
    )


def write_snapshot_manifest(
    upstream_root: Path = SIBLING_PROTOCOL_ROOT,
    vendored_root: Path = VENDORED_PROTOCOL_ROOT,
    upstream_revision: str | None = None,
) -> Path:
    manifest = build_snapshot_manifest(
        upstream_root=upstream_root,
        vendored_root=vendored_root,
        upstream_revision=upstream_revision,
    )
    path = manifest_path_for(vendored_root)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def validate_snapshot_manifest(vendored_root: Path = VENDORED_PROTOCOL_ROOT) -> None:
    ensure_schema_root(vendored_root)
    manifest_path = manifest_path_for(vendored_root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != "pocv3-vendored-schema-snapshot-v1":
        raise ValueError(f"unexpected snapshot manifest schema version: {payload.get('schema_version')}")

    recorded_files = payload.get("files")
    if not isinstance(recorded_files, dict):
        raise ValueError("snapshot manifest files entry must be an object")

    expected_files = set(SCHEMA_FILENAMES)
    if set(recorded_files.keys()) != expected_files:
        raise ValueError("snapshot manifest files must match canonical schema filenames")

    current_files = {
        filename: sha256_file(path)
        for filename, path in required_schema_paths(vendored_root).items()
    }
    if recorded_files != current_files:
        raise ValueError("snapshot manifest hashes do not match vendored schema files")


def refresh_schema_snapshot(
    upstream_root: Path = SIBLING_PROTOCOL_ROOT,
    vendored_root: Path = VENDORED_PROTOCOL_ROOT,
) -> None:
    ensure_schema_root(upstream_root)
    vendored_root.mkdir(parents=True, exist_ok=True)

    for filename, upstream_path in required_schema_paths(upstream_root).items():
        shutil.copy2(upstream_path, vendored_root / filename)
    write_snapshot_manifest(upstream_root=upstream_root, vendored_root=vendored_root)
