"""Canonical JSV helpers aligned to JDVP v1.4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.protocol_core.enums import CONFIDENCE_LEVELS, CORE_FIELD_NAMES  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class JSVRecord:
    """Canonical JSV payload wrapper."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _normalize_timestamp(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if value.endswith("Z"):
        return value
    return value


def _normalize_confidence(confidence: dict[str, Any] | None) -> dict[str, str]:
    if not confidence:
        return {}
    normalized: dict[str, str] = {}
    for name in CORE_FIELD_NAMES:
        level = confidence.get(name)
        if level is None:
            continue
        level_str = str(level)
        if level_str not in CONFIDENCE_LEVELS:
            raise ValueError(f"invalid confidence level for {name}: {level_str}")
        normalized[name] = level_str
    return normalized


def build_jsv(
    *,
    interaction_id: str,
    turn_number: int,
    timestamp: str | None,
    judgment_holder: str,
    delegation_awareness: str,
    cognitive_engagement: str,
    information_seeking: str,
    confidence: dict[str, Any] | None = None,
    context_module: str = "general",
    extensions: dict[str, Any] | None = None,
) -> JSVRecord:
    payload: dict[str, Any] = {
        "timestamp": _normalize_timestamp(timestamp),
        "interaction_id": interaction_id,
        "turn_number": int(turn_number),
        "judgment_holder": judgment_holder,
        "delegation_awareness": delegation_awareness,
        "cognitive_engagement": cognitive_engagement,
        "information_seeking": information_seeking,
        "context_module": context_module,
    }
    normalized_confidence = _normalize_confidence(confidence)
    if normalized_confidence:
        payload["confidence"] = normalized_confidence
    if context_module != "general" and extensions is not None:
        payload["extensions"] = extensions
    return JSVRecord(payload=payload)


def build_jsv_from_hint(
    *,
    interaction_id: str,
    turn_number: int,
    timestamp: str | None,
    context_module: str,
    hint: dict[str, Any],
) -> JSVRecord:
    return build_jsv(
        interaction_id=interaction_id,
        turn_number=turn_number,
        timestamp=timestamp,
        judgment_holder=str(hint["judgment_holder"]),
        delegation_awareness=str(hint["delegation_awareness"]),
        cognitive_engagement=str(hint["cognitive_engagement"]),
        information_seeking=str(hint["information_seeking"]),
        confidence=hint.get("confidence"),
        context_module=context_module,
        extensions=hint.get("extensions"),
    )
