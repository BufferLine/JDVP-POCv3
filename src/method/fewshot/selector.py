"""Selection policy for few-shot examples."""

from __future__ import annotations

from collections import defaultdict
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
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        judgment_holder = str(example.get("jsv_hint", {}).get("judgment_holder", "Unknown"))
        buckets[judgment_holder].append(example)

    ordered_examples: list[dict[str, Any]] = []
    bucket_names = sorted(buckets.keys())
    while bucket_names and len(ordered_examples) < max_examples:
        next_round: list[str] = []
        for bucket_name in bucket_names:
            bucket = buckets[bucket_name]
            if bucket and len(ordered_examples) < max_examples:
                ordered_examples.append(bucket.pop(0))
            if bucket:
                next_round.append(bucket_name)
        bucket_names = next_round
    return ordered_examples
