"""Validation helpers for raw interaction payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema.validators import Draft202012Validator


POCV3_ROOT = Path(__file__).resolve().parents[2]
RAW_INTERACTION_SCHEMA_PATH = POCV3_ROOT / "src" / "contracts" / "raw_interaction.schema.json"


class RawInteractionValidator:
    """Validate raw interaction records against the local contract."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self.schema_path = schema_path or RAW_INTERACTION_SCHEMA_PATH
        with self.schema_path.open("r", encoding="utf-8") as handle:
            self._schema = json.load(handle)
        Draft202012Validator.check_schema(self._schema)
        self._validator = Draft202012Validator(self._schema)

    def validate(self, payload: dict[str, Any]) -> None:
        self._validator.validate(payload)
