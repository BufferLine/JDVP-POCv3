"""Validation against canonical JDVP schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import FormatChecker
from referencing import Registry, Resource


POCV3_ROOT = Path(__file__).resolve().parents[2]
VENDORED_PROTOCOL_ROOT = POCV3_ROOT / "vendor" / "JDVP-protocol" / "v1" / "schemas"
SIBLING_PROTOCOL_ROOT = POCV3_ROOT.parent / "JDVP-protocol" / "v1" / "schemas"


class CanonicalSchemaValidator:
    """Validate generated artifacts against canonical JDVP schemas."""

    def __init__(self, protocol_schema_root: Path | None = None) -> None:
        self.protocol_schema_root = protocol_schema_root or self._resolve_default_root()
        self._schemas = self._load_schemas()

    def _resolve_default_root(self) -> Path:
        if VENDORED_PROTOCOL_ROOT.is_dir():
            return VENDORED_PROTOCOL_ROOT
        return SIBLING_PROTOCOL_ROOT

    def _load_schemas(self) -> dict[str, dict[str, Any]]:
        schema_files = {
            "jsv": self.protocol_schema_root / "jsv-schema.json",
            "dv": self.protocol_schema_root / "dv-schema.json",
            "trajectory": self.protocol_schema_root / "trajectory-schema.json",
        }
        schemas: dict[str, dict[str, Any]] = {}
        for name, path in schema_files.items():
            with path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            Draft202012Validator.check_schema(schema)
            schemas[name] = schema
        return schemas

    def validate_jsv(self, payload: dict[str, Any]) -> None:
        Draft202012Validator(self._schemas["jsv"], format_checker=FormatChecker()).validate(payload)

    def validate_dv(self, payload: dict[str, Any]) -> None:
        Draft202012Validator(self._schemas["dv"], format_checker=FormatChecker()).validate(payload)

    def validate_trajectory(self, payload: dict[str, Any]) -> None:
        patched = json.loads(json.dumps(self._schemas["trajectory"]))
        patched["properties"]["vectors"]["items"]["$ref"] = "https://jdvp.local/schemas/dv-schema.json"
        registry = Registry().with_resource(
            "https://jdvp.local/schemas/dv-schema.json",
            Resource.from_contents(self._schemas["dv"]),
        )
        Draft202012Validator(patched, registry=registry, format_checker=FormatChecker()).validate(payload)
