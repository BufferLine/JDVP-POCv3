"""Direct level-delta derivation aligned to JDVP v1.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.protocol_core.enums import normalize_core_level


@dataclass(frozen=True)
class DVRecord:
    """Canonical DV payload wrapper."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _delta(field_name: str, before: object, after: object) -> int | None:
    before_level = normalize_core_level(field_name, before)
    after_level = normalize_core_level(field_name, after)
    if field_name == "judgment_holder" and (before_level is None or after_level is None):
        return None
    assert before_level is not None and after_level is not None
    return after_level - before_level


def build_dv(
    before: dict[str, Any],
    after: dict[str, Any],
) -> DVRecord:
    if before["interaction_id"] != after["interaction_id"]:
        raise ValueError("cannot derive DV across different interaction_id values")
    if after["turn_number"] <= before["turn_number"]:
        raise ValueError("DV requires increasing turns")

    context_module = after.get("context_module", before.get("context_module", "general"))
    payload: dict[str, Any] = {
        "interaction_id": before["interaction_id"],
        "from_turn": before["turn_number"],
        "to_turn": after["turn_number"],
        "delta_judgment_holder": _delta(
            "judgment_holder",
            before["judgment_holder"],
            after["judgment_holder"],
        ),
        "delta_delegation_awareness": _delta(
            "delegation_awareness",
            before["delegation_awareness"],
            after["delegation_awareness"],
        ),
        "delta_cognitive_engagement": _delta(
            "cognitive_engagement",
            before["cognitive_engagement"],
            after["cognitive_engagement"],
        ),
        "delta_information_seeking": _delta(
            "information_seeking",
            before["information_seeking"],
            after["information_seeking"],
        ),
        "context_module": context_module,
    }
    # Non-general contexts require domain-specific extension deltas (e.g.
    # delta_risk_ownership for financial).  Until extension ordinal maps are
    # implemented, omitting the key is safer than emitting an empty dict that
    # will fail schema validation.
    return DVRecord(payload=payload)


def build_dv_sequence(jsv_sequence: list[dict[str, Any]]) -> list[DVRecord]:
    if len(jsv_sequence) < 2:
        return []
    return [
        build_dv(jsv_sequence[idx], jsv_sequence[idx + 1])
        for idx in range(len(jsv_sequence) - 1)
    ]
