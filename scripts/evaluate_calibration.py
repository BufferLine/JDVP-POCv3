#!/usr/bin/env python3
"""Evaluate one stored observer run against an adjudicated human calibration set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.calibration import build_calibration_report, load_calibration_dataset


def _load_predictions(run_dir: Path) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    extracts = run_dir / "extracts" / manifest["track_name"] / "extracts.jsonl"
    predictions: dict[tuple[str, int], dict] = {}
    metadata: list[dict] = []
    for line in extracts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        predictions[(row["interaction_id"], row["turn_number"])] = row["jsv_hint"]
        track_metadata = dict(row.get("track_metadata", {}))
        metadata.append(track_metadata)
    return predictions, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest, rows = load_calibration_dataset(args.dataset_root)
    predictions, metadata = _load_predictions(args.run_dir)
    report = build_calibration_report(
        manifest=manifest,
        rows=rows,
        predictions=predictions,
        run_metadata=metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration report written: {args.output}")


if __name__ == "__main__":
    main()
