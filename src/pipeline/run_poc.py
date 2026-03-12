"""Smoke pipeline for M1 core validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.protocol_core.dv_ordinal import build_dv_sequence
from src.protocol_core.jsv_types import build_jsv_from_hint
from src.protocol_core.schema_validate import CanonicalSchemaValidator
from src.protocol_core.trajectory import build_trajectory


def _load_input(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_jsv_sequence(raw_interaction: dict[str, Any]) -> list[dict[str, Any]]:
    interaction_id = str(raw_interaction["interaction_id"])
    context_module = str(raw_interaction.get("context_module", "general"))
    jsv_sequence: list[dict[str, Any]] = []
    for turn in raw_interaction["turns"]:
        meta = turn.get("meta", {})
        if "jsv_hint" not in meta:
            raise ValueError(f"turn {turn['turn_number']} missing meta.jsv_hint")
        jsv = build_jsv_from_hint(
            interaction_id=interaction_id,
            turn_number=int(turn["turn_number"]),
            timestamp=meta.get("timestamp") or turn.get("timestamp"),
            context_module=context_module,
            hint=meta["jsv_hint"],
        )
        jsv_sequence.append(jsv.to_dict())
    return jsv_sequence


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_poc(input_path: Path, run_id: str, output_root: Path, protocol_schema_root: Path | None = None) -> Path:
    raw_interaction = _load_input(input_path)
    validator = CanonicalSchemaValidator(protocol_schema_root=protocol_schema_root)

    jsv_sequence = _build_jsv_sequence(raw_interaction)
    dv_sequence = [record.to_dict() for record in build_dv_sequence(jsv_sequence)]
    trajectory = build_trajectory(str(raw_interaction["interaction_id"]), dv_sequence).to_dict()

    for jsv in jsv_sequence:
        validator.validate_jsv(jsv)
    for dv in dv_sequence:
        validator.validate_dv(dv)
    validator.validate_trajectory(trajectory)

    run_dir = output_root / run_id
    _write_json(run_dir / "input" / input_path.name, raw_interaction)
    for jsv in jsv_sequence:
        _write_json(run_dir / "canonical" / "jsv" / f"turn-{jsv['turn_number']}.json", jsv)
    for dv in dv_sequence:
        _write_json(run_dir / "canonical" / "dv" / f"{dv['from_turn']}-{dv['to_turn']}.json", dv)
    _write_json(run_dir / "canonical" / "trajectory.json", trajectory)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "input_path": str(input_path),
            "interaction_id": raw_interaction["interaction_id"],
            "jsv_count": len(jsv_sequence),
            "dv_count": len(dv_sequence),
            "output_root": str(run_dir),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the POCv3 M1 smoke pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--protocol-schema-root", type=Path, default=None)
    args = parser.parse_args()

    run_dir = run_poc(
        input_path=args.input,
        run_id=args.run_id,
        output_root=args.output_root,
        protocol_schema_root=args.protocol_schema_root,
    )
    print(f"Run written: {run_dir}")


if __name__ == "__main__":
    main()
