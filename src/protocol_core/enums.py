"""Shared values for the JDVP v1.5 level-based core fields.

Canonical core values are integer readings on the inclusive 0--10 scale.
Legacy v1.4 categorical values are retained solely to migrate existing
fixtures and resumable extracts into canonical v1.5 output.
"""

from __future__ import annotations

# Ordered tuple of the four core JSV classification field names.
CORE_FIELD_NAMES: tuple[str, ...] = (
    "judgment_holder",
    "delegation_awareness",
    "cognitive_engagement",
    "information_seeking",
)

CORE_LEVEL_MIN = 0
CORE_LEVEL_MAX = 10

# Midpoints specified by the v1.5 migration guidance. They support migration
# of stored v1.4 data; new tracks must emit numeric levels directly.
LEGACY_CATEGORY_MIDPOINTS: dict[str, dict[str, int | None]] = {
    "judgment_holder": {"Human": 2, "Shared": 5, "AI": 9, "Undefined": None},
    "delegation_awareness": {"Explicit": 2, "Implicit": 5, "Absent": 9},
    "cognitive_engagement": {"Active": 2, "Reactive": 5, "Passive": 9},
    "information_seeking": {"Active": 2, "Passive": 5, "None": 9},
}


def normalize_core_level(field_name: str, value: object) -> int | None:
    """Return a canonical v1.5 level, accepting legacy input for migration.

    Only ``judgment_holder`` may be null. ``bool`` is rejected even though it
    is an ``int`` subclass because it is not a meaningful level reading.
    """
    if field_name not in CORE_FIELD_NAMES:
        raise ValueError(f"unknown core field: {field_name}")
    if value is None:
        if field_name == "judgment_holder":
            return None
        raise ValueError(f"{field_name} must be an integer level")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer level")
    if isinstance(value, int):
        if CORE_LEVEL_MIN <= value <= CORE_LEVEL_MAX:
            return value
        raise ValueError(f"{field_name} level must be between {CORE_LEVEL_MIN} and {CORE_LEVEL_MAX}")
    if isinstance(value, str) and value.isdigit():
        return normalize_core_level(field_name, int(value))
    if isinstance(value, str) and value in LEGACY_CATEGORY_MIDPOINTS[field_name]:
        return LEGACY_CATEGORY_MIDPOINTS[field_name][value]
    raise ValueError(f"invalid {field_name} level: {value!r}")

# Valid confidence level values.
CONFIDENCE_LEVELS: frozenset[str] = frozenset({"high", "medium", "low"})
# Alias used in llm_response normalization.
CONFIDENCE_ENUM: frozenset[str] = CONFIDENCE_LEVELS
