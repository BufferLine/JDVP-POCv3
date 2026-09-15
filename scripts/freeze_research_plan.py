#!/usr/bin/env python3
"""Validate and freeze an R0 experiment plan; no provider calls or training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.plan import freeze_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = freeze_plan(args.config, args.output)
    print(f"Frozen research plan: {args.output} (sha256={frozen['plan_sha256']})")


if __name__ == "__main__":
    main()
