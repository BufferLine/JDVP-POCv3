"""Ordinal DV derivation aligned to JDVP v1.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.protocol_core.enums import CORE_ORDINALS  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class DVRecord:
    """Canonical DV payload wrapper."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _delta(field_name: str, before: str, after: str) -> float | None:
    ordinal_map = CORE_ORDINALS[field_name]
    if field_name == "judgment_holder" and (before == "Undefined" or after == "Undefined"):
        return None
    if before not in ordinal_map or after not in ordinal_map:
        raise ValueError(f"invalid ordinal transition for {field_name}: {before} -> {after}")
    scale = len(ordinal_map) - 1
    return (ordinal_map[after] - ordinal_map[before]) / scale


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
            str(before["judgment_holder"]),
            str(after["judgment_holder"]),
        ),
        "delta_delegation_awareness": _delta(
            "delegation_awareness",
            str(before["delegation_awareness"]),
            str(after["delegation_awareness"]),
        ),
        "delta_cognitive_engagement": _delta(
            "cognitive_engagement",
            str(before["cognitive_engagement"]),
            str(after["cognitive_engagement"]),
        ),
        "delta_information_seeking": _delta(
            "information_seeking",
            str(before["information_seeking"]),
            str(after["information_seeking"]),
        ),
        "context_module": context_module,
    }
    if context_module != "general" and "extensions" in before and "extensions" in after:
        payload["extensions"] = {}
    return DVRecord(payload=payload)


def build_dv_sequence(jsv_sequence: list[dict[str, Any]]) -> list[DVRecord]:
    if len(jsv_sequence) < 2:
        return []
    return [
        build_dv(jsv_sequence[idx], jsv_sequence[idx + 1])
        for idx in range(len(jsv_sequence) - 1)
    ]
