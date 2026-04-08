#!/usr/bin/env python3
"""Export trained Ridge regressors to pickle for production use.

Run once at build time:
    python3 export_regressors.py --model-path /app/models/embedding \
        --silver-path /app/data/silver --interactions-path /app/data/interactions \
        --output /app/regressors.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]
DA_WEIGHTS = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570, "information_dependency": 0.268}


def load_training_data(silver_path: Path, interactions_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    interaction_texts: dict[str, dict[int, dict]] = {}
    for f in sorted(interactions_path.glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        interaction_texts[iid] = {t["turn_number"]: t for t in d["turns"]}

    texts = []
    labels = {a: [] for a in AXES}

    for conv_dir in sorted(silver_path.iterdir()):
        if not conv_dir.is_dir():
            continue
        iid = conv_dir.name
        if iid not in interaction_texts:
            continue

        conv_turns = interaction_texts[iid]
        history: list[str] = []

        for tf in sorted(conv_dir.glob("turn-*.json")):
            try:
                label = json.loads(tf.read_text())
            except (json.JSONDecodeError, ValueError):
                continue
            if not all(a in label for a in AXES):
                continue
            tn = int(tf.stem.replace("turn-", ""))
            if tn not in conv_turns:
                continue

            human_input = conv_turns[tn].get("human_input", "")
            history_text = ""
            if history:
                history_text = " | ".join(h[-200:] for h in history[-3:]) + " || "

            texts.append(history_text + f"Human: {human_input}")
            for a in AXES:
                labels[a].append(float(label[a]))
            history.append(human_input)

    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--silver-path", required=True)
    parser.add_argument("--interactions-path", required=True)
    parser.add_argument("--output", default="regressors.pkl")
    args = parser.parse_args()

    print("Loading training data...")
    texts, labels = load_training_data(Path(args.silver_path), Path(args.interactions_path))
    print(f"  {len(texts)} turns loaded")

    print(f"Loading embedding model from {args.model_path}...")
    embedder = SentenceTransformer(args.model_path)

    print("Embedding training data...")
    X = embedder.encode(texts, show_progress_bar=True, batch_size=64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = normalize(X)

    print("Training Ridge regressors...")
    regressors = {}
    for axis in AXES:
        y = np.array(labels[axis])
        reg = Ridge(alpha=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reg.fit(X, y)
        regressors[axis] = reg
        print(f"  {axis}: coef range [{reg.coef_.min():.3f}, {reg.coef_.max():.3f}]")

    output = {
        "regressors": regressors,
        "axes": AXES,
        "da_weights": DA_WEIGHTS,
        "n_training_turns": len(texts),
    }
    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved to {args.output} ({Path(args.output).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
