#!/usr/bin/env python3
"""Train persisted four-axis embedding calibration heads from labeled interactions.

The supplied labels determine the interpretation of the resulting scores. This
script records their provenance and a grouped-by-interaction holdout result in
the artifact; it does not turn silver or synthetic labels into human evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method.tracks.embedding_calibrated import ARTIFACT_SCHEMA_VERSION
from src.method.tracks.embedding_screen import _turn_text, _unit_rows
from src.protocol_core.enums import CORE_FIELD_NAMES, normalize_core_level


def load_labeled_turns(dataset_root: Path, *, exclude_contested: bool) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "interactions").glob("*.json")):
        interaction = json.loads(path.read_text(encoding="utf-8"))
        for turn in interaction["turns"]:
            meta = turn.get("meta", {})
            labels = meta.get("gold_label")
            if not labels or (exclude_contested and meta.get("contested", False)):
                continue
            if not all(field_name in labels for field_name in CORE_FIELD_NAMES):
                continue
            normalized_labels = {
                field_name: normalize_core_level(field_name, labels[field_name])
                for field_name in CORE_FIELD_NAMES
            }
            if any(value is None for value in normalized_labels.values()):
                continue
            turns.append({
                "interaction_id": interaction["interaction_id"],
                "turn_number": turn["turn_number"],
                "text": _turn_text(turn["human_input"], turn["ai_response"]),
                "labels": normalized_labels,
            })
    if not turns:
        raise ValueError(f"no labeled turns found under {dataset_root}")
    return turns


def is_holdout(interaction_id: str, *, test_ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{interaction_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < test_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="jdvp-embedding-calibrated-v1")
    parser.add_argument("--label-source", default="synthetic_or_silver")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=0.25)
    parser.add_argument("--include-contested", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")

    turns = load_labeled_turns(args.dataset_root, exclude_contested=not args.include_contested)
    holdout_flags = [
        is_holdout(turn["interaction_id"], test_ratio=args.test_ratio, seed=args.seed)
        for turn in turns
    ]
    train_turns = [turn for turn, is_test in zip(turns, holdout_flags) if not is_test]
    test_turns = [turn for turn, is_test in zip(turns, holdout_flags) if is_test]
    if not train_turns or not test_turns:
        raise ValueError("grouped split produced an empty train or holdout set")

    embedder = SentenceTransformer(str(args.embedding_model), local_files_only=True)
    vectors = _unit_rows(embedder.encode([turn["text"] for turn in turns], show_progress_bar=True, batch_size=64))
    train_indices = [index for index, is_test in enumerate(holdout_flags) if not is_test]
    test_indices = [index for index, is_test in enumerate(holdout_flags) if is_test]
    x_train, x_test = vectors[train_indices], vectors[test_indices]

    heads: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for field_name in CORE_FIELD_NAMES:
        y_train = np.asarray([turn["labels"][field_name] for turn in train_turns], dtype=float)
        y_test = np.asarray([turn["labels"][field_name] for turn in test_turns], dtype=float)
        regressor = Ridge(alpha=args.ridge_alpha).fit(x_train, y_train)
        predicted = np.clip(regressor.predict(x_test), 0, 10)
        rounded = np.rint(predicted)
        heads[field_name] = {
            "coef": [round(float(value), 10) for value in regressor.coef_],
            "intercept": round(float(regressor.intercept_), 10),
        }
        metrics[field_name] = {
            "mae": round(float(mean_absolute_error(y_test, predicted)), 4),
            "rounded_exact_agreement": round(float(np.mean(rounded == y_test)), 4),
        }

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": args.model_id,
        "embedding_model_path": str(args.embedding_model.resolve()),
        "label_source": args.label_source,
        "dataset_root": str(args.dataset_root.resolve()),
        "training_policy": {
            "exclude_contested": not args.include_contested,
            "grouped_by": "interaction_id",
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "canonical_level_mapping": "legacy category labels normalized to JDVP v1.5 integer levels",
        },
        "training_counts": {
            "turns_total": len(turns),
            "turns_train": len(train_turns),
            "turns_holdout": len(test_turns),
            "interactions_train": len({turn['interaction_id'] for turn in train_turns}),
            "interactions_holdout": len({turn['interaction_id'] for turn in test_turns}),
        },
        "holdout_metrics": metrics,
        "heads": heads,
        "interpretation_boundary": "Scores inherit the declared label source and are not human-validated critical-thinking scores.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Embedding calibrator written: {args.output}")
    print(json.dumps({"training_counts": artifact["training_counts"], "holdout_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
