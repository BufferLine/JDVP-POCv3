"""Normalization for LLM observer responses."""

from __future__ import annotations

import json
import re
from typing import Any

from src.protocol_core.enums import CONFIDENCE_ENUM, CORE_ENUMS  # noqa: F401  (re-exported)
SMART_QUOTES = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


class LLMNormalizationError(ValueError):
    """Raised when an LLM response cannot be normalized safely."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        attempt_responses: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.attempt_responses = attempt_responses or ([] if raw_response is None else [raw_response])


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _find_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_code_fences(text)
    candidate = _find_first_json_object(stripped) or stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        normalized_candidate = candidate.translate(SMART_QUOTES)
        if normalized_candidate != candidate:
            try:
                parsed = json.loads(normalized_candidate)
            except json.JSONDecodeError:
                salvaged = _salvage_partial_payload(normalized_candidate)
                if salvaged is not None:
                    return salvaged
                raise LLMNormalizationError(f"invalid JSON response: {exc}") from exc
        else:
            salvaged = _salvage_partial_payload(candidate)
            if salvaged is not None:
                return salvaged
            raise LLMNormalizationError(f"invalid JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMNormalizationError("LLM response must be a JSON object")
    return parsed


def _extract_string_field(text: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _salvage_partial_payload(text: str) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for field_name in CORE_ENUMS:
        value = _extract_string_field(text, field_name)
        if value is None or value not in CORE_ENUMS[field_name]:
            return None
        payload[field_name] = value

    confidence: dict[str, str] = {}
    for field_name in CORE_ENUMS:
        pattern = rf'"confidence"\s*:\s*\{{.*?"{re.escape(field_name)}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, re.DOTALL)
        if not match or match.group(1) not in CONFIDENCE_ENUM:
            return None
        confidence[field_name] = match.group(1)
    payload["confidence"] = confidence

    evidence_text = None
    evidence_category = None
    text_match = re.search(r'"evidence_spans"\s*:\s*\[\s*\{\s*"text"\s*:\s*"(.+?)(?="\s*,\s*"category")', text, re.DOTALL)
    category_match = re.search(r'"category"\s*:\s*"([^"]+)"', text)
    if text_match:
        evidence_text = text_match.group(1).strip()
    if category_match:
        evidence_category = category_match.group(1).strip()
    if not evidence_text:
        evidence_text = "salvaged_evidence"
    if not evidence_category:
        evidence_category = "salvaged_evidence"
    payload["evidence_spans"] = [{"text": evidence_text, "category": evidence_category}]

    observer_notes = _extract_string_field(text, "observer_notes")
    if observer_notes is not None:
        payload["observer_notes"] = observer_notes
    observer_confidence = re.search(r'"observer_confidence"\s*:\s*([0-9.]+)', text)
    if observer_confidence:
        payload["observer_confidence"] = float(observer_confidence.group(1))
    return payload


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
