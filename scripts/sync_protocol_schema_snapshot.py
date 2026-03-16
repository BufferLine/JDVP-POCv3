#!/usr/bin/env python3
"""Refresh vendored JDVP schemas from the sibling protocol repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocol_core.schema_sync import compare_schema_roots, refresh_schema_snapshot
from src.protocol_core.schema_validate import SIBLING_PROTOCOL_ROOT, VENDORED_PROTOCOL_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=SIBLING_PROTOCOL_ROOT)
    parser.add_argument("--vendored-root", type=Path, default=VENDORED_PROTOCOL_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    refresh_schema_snapshot(
        upstream_root=args.upstream_root,
        vendored_root=args.vendored_root,
    )
    diffs = compare_schema_roots(
        upstream_root=args.upstream_root,
        vendored_root=args.vendored_root,
    )
    if diffs:
        print("FAIL: vendored schemas still differ after refresh")
        return 1

    print(
        "OK: refreshed vendored schemas from upstream "
        f"({args.upstream_root} -> {args.vendored_root})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
