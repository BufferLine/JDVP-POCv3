"""Selection policy for few-shot examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.run_storage import read_json


def load_fewshot_pack(pack_path: Path) -> dict[str, Any]:
    return read_json(pack_path)


def select_examples(
    *,
    pack: dict[str, Any],
    target_interaction_id: str,
    context_module: str,
    max_examples: int,
) -> list[dict[str, Any]]:
    examples = [
        example
        for example in pack["examples"]
        if example["interaction_id"] != target_interaction_id and example["context_module"] == context_module
    ]
    return examples[:max_examples]
