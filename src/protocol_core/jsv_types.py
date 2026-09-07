"""Canonical JSV helpers aligned to JDVP v1.5."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.protocol_core.enums import (  # noqa: F401  (re-exported)
    CONFIDENCE_LEVELS,
    CORE_FIELD_NAMES,
    normalize_core_level,
)

logger = logging.getLogger(__name__)


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
    judgment_holder: int | str | None,
    delegation_awareness: int | str,
    cognitive_engagement: int | str,
    information_seeking: int | str,
    confidence: dict[str, Any] | None = None,
    context_module: str = "general",
    extensions: dict[str, Any] | None = None,
) -> JSVRecord:
    payload: dict[str, Any] = {
        "timestamp": _normalize_timestamp(timestamp),
        "interaction_id": interaction_id,
        "turn_number": int(turn_number),
        "judgment_holder": normalize_core_level("judgment_holder", judgment_holder),
        "delegation_awareness": normalize_core_level("delegation_awareness", delegation_awareness),
        "cognitive_engagement": normalize_core_level("cognitive_engagement", cognitive_engagement),
        "information_seeking": normalize_core_level("information_seeking", information_seeking),
        "context_module": context_module,
    }
    normalized_confidence = _normalize_confidence(confidence)
    if normalized_confidence:
        payload["confidence"] = normalized_confidence
    if context_module != "general":
        if extensions is not None:
            payload["extensions"] = extensions
        else:
            logger.warning(
                "build_jsv: non-general context_module=%r but no extensions provided; "
                "resulting JSV may fail schema validation",
                context_module,
            )
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
        judgment_holder=hint["judgment_holder"],
        delegation_awareness=hint["delegation_awareness"],
        cognitive_engagement=hint["cognitive_engagement"],
        information_seeking=hint["information_seeking"],
        confidence=hint.get("confidence"),
        context_module=context_module,
        extensions=hint.get("extensions"),
    )
