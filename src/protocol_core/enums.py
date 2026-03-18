"""Shared enum values and ordinals for JDVP v1.4 protocol fields.

Single source of truth for field names, valid enum values, ordinal mappings,
and confidence levels used across protocol_core and extraction tracks.
"""

from __future__ import annotations

# Ordered tuple of the four core JSV classification field names.
CORE_FIELD_NAMES: tuple[str, ...] = (
    "judgment_holder",
    "delegation_awareness",
    "cognitive_engagement",
    "information_seeking",
)

# Valid enum values for each core field.
# judgment_holder includes "Undefined" as a valid LLM-response value even though
# it does not appear in the ordinal map (ordinal transitions involving Undefined
# are treated as None by dv_ordinal._delta).
CORE_ENUMS: dict[str, frozenset[str]] = {
    "judgment_holder": frozenset({"Human", "Shared", "AI", "Undefined"}),
    "delegation_awareness": frozenset({"Explicit", "Implicit", "Absent"}),
    "cognitive_engagement": frozenset({"Active", "Reactive", "Passive"}),
    "information_seeking": frozenset({"Active", "Passive", "None"}),
}

# Ordinal position of each enum value within its field.
# "Undefined" is intentionally absent from judgment_holder here; callers that
# need to handle it (e.g. dv_ordinal._delta) do so explicitly.
CORE_ORDINALS: dict[str, dict[str, int]] = {
    "judgment_holder": {"Human": 0, "Shared": 1, "AI": 2},
    "delegation_awareness": {"Explicit": 0, "Implicit": 1, "Absent": 2},
    "cognitive_engagement": {"Active": 0, "Reactive": 1, "Passive": 2},
    "information_seeking": {"Active": 0, "Passive": 1, "None": 2},
}

# Valid confidence level values.
CONFIDENCE_LEVELS: frozenset[str] = frozenset({"high", "medium", "low"})
# Alias used in llm_response normalization.
CONFIDENCE_ENUM: frozenset[str] = CONFIDENCE_LEVELS
