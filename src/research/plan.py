"""Freeze a predeclared research plan and its input bytes without model calls."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from jsonschema.validators import Draft202012Validator

from src.contracts.raw_interaction_validate import RawInteractionValidator

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "contracts/research_plan.schema.json"


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("plan contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot(base: Path, name: str) -> dict[str, str]:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"snapshot path must stay within plan directory: {name}")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"snapshot path escapes plan directory: {name}")
    data = path.read_bytes()
    # Store exact UTF-8 bytes as text, not only a pointer to mutable input files.
    return {"sha256": _digest(data), "text": data.decode("utf-8")}


def build_frozen_plan(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _finite(config)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(config)
    base = config_path.parent
    validator = RawInteractionValidator()
    interactions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    family_splits: dict[str, str] = {}
    split_ids: dict[str, list[str]] = {name: [] for name in ("train", "dev", "test")}
    reserved = set(config["reserved_test_domains"])
    observed_reserved: set[str] = set()
    for item in config["items"]:
        snapshot = _snapshot(base, item["path"])
        raw = json.loads(snapshot["text"])
        validator.validate(raw)
        iid = raw["interaction_id"]
        if iid in seen_ids or snapshot["sha256"] in seen_hashes:
            raise ValueError(f"duplicate interaction or content: {iid}")
        seen_ids.add(iid)
        seen_hashes.add(snapshot["sha256"])
        split, family, domain = item["split"], item["template_family"], item["domain"]
        if family in family_splits and family_splits[family] != split:
            raise ValueError(f"template family crosses splits: {family}")
        family_splits[family] = split
        if domain in reserved:
            if split != "test":
                raise ValueError(f"reserved test domain appears in {split}: {domain}")
            observed_reserved.add(domain)
        split_ids[split].append(iid)
        interactions.append({**item, "interaction_id": iid, "snapshot": snapshot})
    if any(not ids for ids in split_ids.values()):
        raise ValueError("train, dev and test must each contain an interaction")
    if observed_reserved != reserved:
        raise ValueError("reserved test domains must have test examples")
    if config["sampling"]["max_items"] > len(split_ids["train"]) + len(split_ids["dev"]):
        raise ValueError("sampling max_items exceeds train/dev pool; test data is evaluation-only")
    rubric = _snapshot(base, config["rubric_path"])
    if not rubric["text"].strip():
        raise ValueError("rubric must not be empty")
    criteria = _snapshot(base, config["external_criteria"]["source_path"])
    if not criteria["text"].strip() or criteria["sha256"] in seen_hashes | {rubric["sha256"]}:
        raise ValueError("external criteria evidence must be nonempty and separate from input records and rubric")
    body = {
        "schema_version": "jdvp-frozen-research-plan-v1",
        "plan": config,
        "rubric": rubric,
        "external_evidence": criteria,
        "interactions": interactions,
        "split_sha256": _digest(_canonical({k: sorted(v) for k, v in split_ids.items()})),
        "schema_sha256": _digest(_canonical(schema)),
        "limitations": [
            "Authorization, family/domain labels, and encoder exposure are declared, not independently verified.",
            "External evidence independence requires semantic review; byte separation alone cannot establish it.",
            "Freezing does not execute sampling, spend enforcement, evaluation or artifact promotion.",
        ],
    }
    return {**body, "plan_sha256": _digest(_canonical(body))}


def freeze_plan(config_path: Path, output_path: Path) -> dict[str, Any]:
    """Publish a complete manifest atomically; never replace an existing freeze."""
    frozen = build_frozen_plan(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory hard link publishes atomically with no overwrite race.
    fd, temporary = tempfile.mkstemp(dir=output_path.parent, prefix=".research-plan-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(frozen) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    finally:
        os.unlink(temporary)
    return frozen
