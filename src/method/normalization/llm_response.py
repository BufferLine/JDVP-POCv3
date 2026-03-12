"""Normalization for LLM observer responses."""

from __future__ import annotations

import json
from typing import Any


CORE_ENUMS: dict[str, set[str]] = {
    "judgment_holder": {"Human", "Shared", "AI", "Undefined"},
    "delegation_awareness": {"Explicit", "Implicit", "Absent"},
    "cognitive_engagement": {"Active", "Reactive", "Passive"},
    "information_seeking": {"Active", "Passive", "None"},
}
CONFIDENCE_ENUM = {"high", "medium", "low"}


class LLMNormalizationError(ValueError):
    """Raised when an LLM response cannot be normalized safely."""


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMNormalizationError(f"invalid JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMNormalizationError("LLM response must be a JSON object")
    return parsed


def _normalize_confidence(payload: dict[str, Any]) -> dict[str, str]:
    confidence = payload.get("confidence")
    if not isinstance(confidence, dict):
        raise LLMNormalizationError("confidence must be an object")
    normalized: dict[str, str] = {}
    for field_name in CORE_ENUMS:
        raw = confidence.get(field_name)
        if not isinstance(raw, str) or raw not in CONFIDENCE_ENUM:
            raise LLMNormalizationError(f"invalid confidence for {field_name}: {raw}")
        normalized[field_name] = raw
    return normalized


def _normalize_evidence_spans(payload: dict[str, Any]) -> list[dict[str, str]]:
    evidence_spans = payload.get("evidence_spans", [])
    if not isinstance(evidence_spans, list):
        raise LLMNormalizationError("evidence_spans must be an array")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(evidence_spans):
        if not isinstance(item, dict):
            raise LLMNormalizationError(f"evidence span at index {index} must be an object")
        text = item.get("text")
        category = item.get("category")
        note = item.get("note")
        if not isinstance(text, str) or not text:
            raise LLMNormalizationError(f"evidence span at index {index} missing text")
        if not isinstance(category, str) or not category:
            raise LLMNormalizationError(f"evidence span at index {index} missing category")
        normalized_item = {"text": text, "category": category}
        if isinstance(note, str) and note:
            normalized_item["note"] = note
        normalized.append(normalized_item)
    if not normalized:
        raise LLMNormalizationError("at least one evidence span is required")
    return normalized


def normalize_llm_response(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    normalized_hint: dict[str, Any] = {}
    for field_name, allowed in CORE_ENUMS.items():
        raw = payload.get(field_name)
        if not isinstance(raw, str) or raw not in allowed:
            raise LLMNormalizationError(f"invalid value for {field_name}: {raw}")
        normalized_hint[field_name] = raw

    normalized_hint["confidence"] = _normalize_confidence(payload)
    evidence_spans = _normalize_evidence_spans(payload)

    observer_notes = payload.get("observer_notes", "")
    if observer_notes and not isinstance(observer_notes, str):
        raise LLMNormalizationError("observer_notes must be a string when present")

    observer_confidence = payload.get("observer_confidence")
    if observer_confidence is not None:
        if not isinstance(observer_confidence, (int, float)) or not (0.0 <= float(observer_confidence) <= 1.0):
            raise LLMNormalizationError("observer_confidence must be between 0.0 and 1.0")

    normalized: dict[str, Any] = {
        "jsv_hint": normalized_hint,
        "evidence_spans": evidence_spans,
        "observer_notes": observer_notes if isinstance(observer_notes, str) else "",
    }
    if observer_confidence is not None:
        normalized["observer_confidence"] = float(observer_confidence)
    return normalized
