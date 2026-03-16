#!/usr/bin/env python3
"""Train and persist the cheap ML baseline from a few-shot pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method.tracks.cheap_ml_baseline import CheapMLBaselineTrack


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cheap ML baseline from a few-shot pack")
    parser.add_argument("--fewshot-pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default="naive-bayes-v1")
    args = parser.parse_args()

    track = CheapMLBaselineTrack(
        fewshot_pack_path=args.fewshot_pack,
        model_id=args.model_id,
    )
    output_path = track.write_artifact(args.output)
    print(f"Cheap ML model written: {output_path}")


if __name__ == "__main__":
    main()
