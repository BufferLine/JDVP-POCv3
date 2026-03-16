#!/usr/bin/env python3
"""Verify that AGENTS.md and CLAUDE.md stay synchronized."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = ROOT / "AGENTS.md"
CLAUDE_PATH = ROOT / "CLAUDE.md"


def _body(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[1:]


def main() -> int:
    if not AGENTS_PATH.is_file():
        raise FileNotFoundError(f"missing assistant doc: {AGENTS_PATH}")
    if not CLAUDE_PATH.is_file():
        raise FileNotFoundError(f"missing assistant doc: {CLAUDE_PATH}")

    if _body(AGENTS_PATH) != _body(CLAUDE_PATH):
        print("FAIL: AGENTS.md and CLAUDE.md differ below the title line")
        return 1

    print("OK: AGENTS.md and CLAUDE.md are synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
