"""Shared utility functions used across multiple modules."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with Z suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer that lowercases and extracts word tokens."""
    return _TOKEN_RE.findall(text.lower())


def load_json(path: Path) -> Any:
    """Load and parse a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
