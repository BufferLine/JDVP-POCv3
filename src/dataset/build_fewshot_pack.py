"""Build few-shot example packs from stored extract runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.pipeline.run_storage import read_json, write_json


def _load_extract_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def build_fewshot_pack(
    *,
    run_dir: Path,
    output_path: Path,
    max_examples: int = 8,
) -> Path:
    manifest = read_json(run_dir / "manifest.json")
    track_name = str(manifest["track_name"])
    extract_rows = _load_extract_rows(run_dir / "extracts" / track_name / "extracts.jsonl")
    selected_rows = extract_rows[:max_examples]
    pack = {
        "schema_version": "pocv3-fewshot-pack-v1",
        "source_run_id": manifest["run_id"],
        "source_track_name": track_name,
        "interaction_id": manifest["interaction_id"],
        "prompt_contract": {
            "task": "Infer JDVP JSV hints for one human-AI turn and return JSON only.",
            "output_fields": [
                "judgment_holder",
                "delegation_awareness",
                "cognitive_engagement",
                "information_seeking",
                "confidence",
                "evidence_spans",
                "observer_notes",
            ],
        },
        "examples": [
            {
                "interaction_id": row["interaction_id"],
                "turn_number": row["turn_number"],
                "context_module": row["context_module"],
                "human_input": row["human_input"],
                "ai_response": row["ai_response"],
                "jsv_hint": row["jsv_hint"],
                "evidence_spans": row["overlay"].get("evidence_spans", []),
                "observer_notes": row["track_metadata"].get("observer_notes", ""),
            }
            for row in selected_rows
        ],
    }
    write_json(output_path, pack)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a few-shot prompt pack from one stored run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    output_path = build_fewshot_pack(
        run_dir=args.run_dir,
        output_path=args.output,
        max_examples=args.max_examples,
    )
    print(f"Few-shot pack written: {output_path}")


if __name__ == "__main__":
    main()
