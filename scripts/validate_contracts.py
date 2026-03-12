#!/usr/bin/env python3
"""Validate local repository contracts and fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema.validators import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    print(f"OK: {path.relative_to(ROOT)}")


def validate_schema(path: Path) -> dict:
    schema = load_json(path)
    Draft202012Validator.check_schema(schema)
    print(f"OK: schema {path.relative_to(ROOT)}")
    return schema


def validate_instance(instance_path: Path, schema: dict, label: str) -> None:
    Draft202012Validator(schema).validate(load_json(instance_path))
    print(f"OK: {label} {instance_path.relative_to(ROOT)}")


def main() -> None:
    required_docs = [
        ROOT / "README.md",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "IMPLEMENTATION_PLAN.md",
        ROOT / "docs" / "MIGRATION_MATRIX.md",
        ROOT / "docs" / "MIGRATION_NOTES.md",
        ROOT / "docs" / "OPERATIONS.md",
        ROOT / "vendor" / "JDVP-protocol" / "v1" / "schemas" / "jsv-schema.json",
        ROOT / "vendor" / "JDVP-protocol" / "v1" / "schemas" / "dv-schema.json",
        ROOT / "vendor" / "JDVP-protocol" / "v1" / "schemas" / "trajectory-schema.json",
    ]
    for path in required_docs:
        require_file(path)

    raw_schema_path = ROOT / "src" / "contracts" / "raw_interaction.schema.json"
    overlay_schema_path = ROOT / "src" / "contracts" / "annotation_overlay.schema.json"
    fixture_path = ROOT / "data" / "fixtures" / "sample_interaction.json"

    raw_schema = validate_schema(raw_schema_path)
    validate_schema(overlay_schema_path)
    validate_instance(fixture_path, raw_schema, "fixture")


if __name__ == "__main__":
    main()
