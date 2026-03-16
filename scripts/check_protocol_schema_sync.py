#!/usr/bin/env python3
"""Check whether vendored JDVP schemas match the sibling protocol repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocol_core.schema_sync import compare_schema_roots, validate_snapshot_manifest
from src.protocol_core.schema_validate import SIBLING_PROTOCOL_ROOT, VENDORED_PROTOCOL_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=SIBLING_PROTOCOL_ROOT)
    parser.add_argument("--vendored-root", type=Path, default=VENDORED_PROTOCOL_ROOT)
    parser.add_argument(
        "--require-upstream",
        action="store_true",
        help="fail when the upstream schema root does not exist locally",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.upstream_root.is_dir():
        if args.require_upstream:
            raise FileNotFoundError(f"missing upstream schema root: {args.upstream_root}")
        print(f"SKIP: upstream schema root not available: {args.upstream_root}")
        return 0

    diffs = compare_schema_roots(
        upstream_root=args.upstream_root,
        vendored_root=args.vendored_root,
    )
    if diffs:
        print("FAIL: vendored schemas differ from upstream")
        for diff in diffs:
            print(
                f"- {diff.filename}: upstream={diff.upstream_sha256} "
                f"vendored={diff.vendored_sha256}"
            )
        return 1

    validate_snapshot_manifest(args.vendored_root)

    print(
        "OK: vendored schemas match upstream "
        f"({args.upstream_root} == {args.vendored_root})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
