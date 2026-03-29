"""Load prompt templates for observer tracks."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = ROOT / "config" / "prompts"

# Map base prompt names to environment variable overrides.
_ENV_OVERRIDES = {
    "llm_observer_system.txt": "JDVP_OBSERVER_SYSTEM_PROMPT",
}


def load_prompt(name: str) -> str:
    env_key = _ENV_OVERRIDES.get(name)
    if env_key:
        override = os.environ.get(env_key)
        if override:
            path = Path(override)
            if path.is_file():
                return path.read_text(encoding="utf-8")
            # Also check relative to PROMPT_DIR
            alt = PROMPT_DIR / override
            if alt.is_file():
                return alt.read_text(encoding="utf-8")
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")
