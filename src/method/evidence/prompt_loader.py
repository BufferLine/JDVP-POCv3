"""Load prompt templates for observer tracks."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = ROOT / "config" / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")
