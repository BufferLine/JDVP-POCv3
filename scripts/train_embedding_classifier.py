#!/usr/bin/env python3
"""Train embedding-based classifiers on silver or gold validation data.

Embeds turn text with sentence-transformers, trains one classifier per JDVP field.

Usage:
    # Train on silver (wildchat real data)
    python3 scripts/train_embedding_classifier.py --source silver

    # Train on gold (synthetic data, original)
    python3 scripts/train_embedding_classifier.py --source gold

    # Train on silver, test on gold (cross-domain)
    python3 scripts/train_embedding_classifier.py --source silver --cross-eval-gold
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = ROOT / "data" / "validation" / "opus-synthetic"
SILVER_ROOT = ROOT / "data" / "silver" / "wildchat-3agent-v1"
FIELDS = ["judgment_holder", "delegation_awareness", "cognitive_engagement", "information_seeking"]
LEVEL_FIELDS = ["judgment_delegation", "delegation_awareness", "cognitive_passivity", "information_dependency"]


def load_gold_turns() -> list[dict]:
    """Load all gold-labeled turns with text."""
    turns = []
    for f in sorted((GOLD_ROOT / "interactions").glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        scenario = "-".join(iid.split("-")[2:-1])
        history = []
        for t in d["turns"]:
            meta = t.get("meta", {})
            gl = meta.get("gold_label", {})
            if not gl:
                continue
            history_text = ""
            if history:
                history_text = " | ".join(f"H: {h[-200:]}" for h in history[-3:]) + " || "
            turn_text = history_text + f"Human: {t['human_input']}"
            turns.append({
                "interaction_id": iid,
                "scenario": scenario,
                "turn_number": t["turn_number"],
                "text": turn_text,
                "label": gl,
            })
            history.append(t["human_input"])
    return turns


def load_silver_turns() -> list[dict]:
    """Load silver-labeled turns from wildchat 3-agent merge."""
    silver_dir = SILVER_ROOT / "silver-labels"
    interactions_dir = ROOT / "data" / "open-data" / "wildchat" / "v1" / "interactions"

    # Load interaction texts
    interaction_texts = {}
    for f in sorted(interactions_dir.glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        interaction_texts[iid] = {t["turn_number"]: t for t in d["turns"]}

    turns = []
    for iid_dir in sorted(silver_dir.iterdir()):
        if not iid_dir.is_dir():
            continue
        iid = iid_dir.name
        if iid not in interaction_texts:
            continue

        history = []
        for tf in sorted(iid_dir.glob("turn-*.json")):
            label_data = json.loads(tf.read_text())
            tn = label_data["turn_number"]
            jsv = label_data.get("jsv_hint", {})

            # Skip if missing required fields
            if not all(f in jsv for f in FIELDS):
                continue

            interaction_turn = interaction_texts[iid].get(tn, {})
            human_input = interaction_turn.get("human_input", label_data.get("human_input", ""))

            history_text = ""
            if history:
                history_text = " | ".join(f"H: {h[-200:]}" for h in history[-3:]) + " || "
            turn_text = history_text + f"Human: {human_input}"

            turns.append({
                "interaction_id": iid,
                "scenario": "wildchat",
                "turn_number": tn,
                "text": turn_text,
                "label": {f: jsv[f] for f in FIELDS},
                "unanimous": label_data.get("unanimous", False),
            })
            history.append(human_input)
    return turns


def load_level_turns(dataset_dir: str | None = None, labels_dir: str | None = None) -> list[dict]:
    """Load level-based (0-10) labels from level experiment outputs."""
    if labels_dir:
        level_base = Path(labels_dir)
    else:
        level_base = ROOT / "data" / "silver" / "wildchat-level-experiment"

    if dataset_dir:
        interactions_dir = Path(dataset_dir) / "interactions"
    else:
        interactions_dir = ROOT / "data" / "open-data" / "wildchat" / "v1" / "interactions"

    # Load interaction texts
    interaction_texts = {}
    for f in sorted(interactions_dir.glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        interaction_texts[iid] = {t["turn_number"]: t for t in d["turns"]}

    # Find agents and merge by averaging
    agent_dirs = [d for d in sorted(level_base.iterdir()) if d.is_dir()]
    agent_names = [d.name for d in agent_dirs]

    # Collect all (iid, tn) -> {agent: label}
    all_labels: dict[tuple[str, int], dict[str, dict]] = {}
    for agent_dir in agent_dirs:
        for iid_dir in sorted(agent_dir.iterdir()):
            if not iid_dir.is_dir():
                continue
            iid = iid_dir.name
            for tf in sorted(iid_dir.glob("turn-*.json")):
                tn = int(tf.stem.replace("turn-", ""))
                label = json.loads(tf.read_text())
                if not all(f in label for f in LEVEL_FIELDS):
                    continue
                key = (iid, tn)
                if key not in all_labels:
                    all_labels[key] = {}
                all_labels[key][agent_dir.name] = label

    turns = []
    for (iid, tn) in sorted(all_labels.keys()):
        agent_labels = all_labels[(iid, tn)]
        if len(agent_labels) < 2:
            continue

        # Average across agents
        avg_label = {}
        for f in LEVEL_FIELDS:
            vals = [al[f] for al in agent_labels.values() if isinstance(al.get(f), (int, float))]
            if vals:
                avg_label[f] = round(sum(vals) / len(vals), 1)

        if len(avg_label) != len(LEVEL_FIELDS):
            continue

        # Get text
        if iid not in interaction_texts or tn not in interaction_texts[iid]:
            continue
        interaction_turn = interaction_texts[iid][tn]
        human_input = interaction_turn.get("human_input", "")

        # Build history
        prior_tns = sorted([t for t in interaction_texts[iid] if t < tn])
        history = [interaction_texts[iid][t]["human_input"] for t in prior_tns[-3:]]
        history_text = ""
        if history:
            history_text = " | ".join(f"H: {h[-200:]}" for h in history) + " || "

        turns.append({
            "interaction_id": iid,
            "scenario": "wildchat",
            "turn_number": tn,
            "text": history_text + f"Human: {human_input}",
            "level_label": avg_label,
            "num_agents": len(agent_labels),
        })

    return turns


def split_by_interaction(turns: list[dict], test_ratio: float = 0.3, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split by interaction_id to prevent leakage."""
    iids = sorted(set(t["interaction_id"] for t in turns))
    rng = random.Random(seed)
    rng.shuffle(iids)
    test_count = max(1, int(len(iids) * test_ratio))
    test_iids = set(iids[:test_count])
    train = [t for t in turns if t["interaction_id"] not in test_iids]
    test = [t for t in turns if t["interaction_id"] in test_iids]
    return train, test


def split_by_scenario(turns: list[dict], test_scenario_count: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split turns by scenario — no scenario leaks between train/test."""
    scenarios = sorted(set(t["scenario"] for t in turns))
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    test_scenarios = set(scenarios[:test_scenario_count])
    train = [t for t in turns if t["scenario"] not in test_scenarios]
    test = [t for t in turns if t["scenario"] in test_scenarios]
    return train, test


def train_and_eval(X_train, y_train, X_test, y_test, field: str, seed: int = 42) -> float:
    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{'='*50}")
    print(f"{field}: {acc:.1%}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train embedding classifiers")
    parser.add_argument("--source", choices=["gold", "silver", "both", "level"], default="silver")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-eval-gold", action="store_true",
                        help="Train on silver, evaluate on gold")
    parser.add_argument("--unanimous-only", action="store_true",
                        help="Only use unanimous silver labels for training")
    args = parser.parse_args()

    # Load data
    if args.source == "level":
        # --- REGRESSION MODE ---
        print("Loading level turns (0-10 scale)...")
        turns = load_level_turns()
        print(f"Total: {len(turns)} turns")
        train_turns, test_turns = split_by_interaction(turns, 0.3, args.seed)
        print(f"Train: {len(train_turns)}, Test: {len(test_turns)}")

        print(f"\nLoading embedding model: {args.model}...")
        embedder = SentenceTransformer(args.model)
        print("Embedding texts...")
        X_train = embedder.encode([t["text"] for t in train_turns], show_progress_bar=True)
        X_test = embedder.encode([t["text"] for t in test_turns], show_progress_bar=True)

        print(f"\n{'#'*50}")
        print(f"# LEVEL REGRESSION (0-10)")
        print(f"{'#'*50}\n")

        results = {}
        for field in LEVEL_FIELDS:
            y_train = np.array([t["level_label"][field] for t in train_turns])
            y_test = np.array([t["level_label"][field] for t in test_turns])

            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            y_pred = np.clip(reg.predict(X_test), 0, 10)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Bucketed accuracy (Low/Mid/High)
            def bucket3(v):
                return 0 if v <= 3 else (1 if v <= 6 else 2)
            y_test_b = [bucket3(v) for v in y_test]
            y_pred_b = [bucket3(v) for v in y_pred]
            bucket_acc = accuracy_score(y_test_b, y_pred_b)

            results[field] = {"mae": mae, "r2": r2, "bucket_acc": bucket_acc}
            print(f"{'='*50}")
            print(f"{field}")
            print(f"  MAE:        {mae:.2f} (avg error on 0-10 scale)")
            print(f"  R²:         {r2:.3f}")
            print(f"  Bucket acc: {bucket_acc:.1%} (Low/Mid/High)")
            print(f"  y_test range: [{y_test.min():.1f}, {y_test.max():.1f}] avg={y_test.mean():.1f}")
            print(f"  y_pred range: [{y_pred.min():.1f}, {y_pred.max():.1f}] avg={y_pred.mean():.1f}")
            print()

        avg_mae = np.mean([r["mae"] for r in results.values()])
        avg_r2 = np.mean([r["r2"] for r in results.values()])
        avg_bucket = np.mean([r["bucket_acc"] for r in results.values()])
        print(f"{'='*50}")
        print(f"OVERALL")
        print(f"  Avg MAE:        {avg_mae:.2f}")
        print(f"  Avg R²:         {avg_r2:.3f}")
        print(f"  Avg Bucket acc: {avg_bucket:.1%}")
        for f, r in results.items():
            print(f"  {f:25s} MAE={r['mae']:.2f}  R²={r['r2']:.3f}  Bucket={r['bucket_acc']:.1%}")

        # Baseline: predict mean
        print(f"\n--- Baseline (predict mean) ---")
        for field in LEVEL_FIELDS:
            y_all = [t["level_label"][field] for t in turns]
            mean_val = np.mean(y_all)
            y_test = [t["level_label"][field] for t in test_turns]
            baseline_mae = np.mean([abs(v - mean_val) for v in y_test])
            print(f"  {field:25s} mean={mean_val:.1f}  MAE={baseline_mae:.2f}")

        return  # skip categorical code path

    if args.source == "gold":
        print("Loading gold turns...")
        turns = load_gold_turns()
        train_turns, test_turns = split_by_scenario(turns, 6, args.seed)
    elif args.source == "silver":
        print("Loading silver turns...")
        turns = load_silver_turns()
        if args.unanimous_only:
            turns = [t for t in turns if t.get("unanimous", False)]
            print(f"  (unanimous only: {len(turns)} turns)")
        train_turns, test_turns = split_by_interaction(turns, 0.3, args.seed)
    else:  # both
        print("Loading silver + gold turns...")
        silver = load_silver_turns()
        gold = load_gold_turns()
        turns = silver + gold
        train_turns, test_turns = split_by_interaction(turns, 0.3, args.seed)

    print(f"Total: {len(turns)}, Train: {len(train_turns)}, Test: {len(test_turns)}")

    # Cross-eval: train on silver, test on gold
    cross_test_turns = None
    if args.cross_eval_gold and args.source == "silver":
        print("Loading gold turns for cross-evaluation...")
        cross_test_turns = load_gold_turns()
        print(f"Cross-eval test: {len(cross_test_turns)} gold turns")

    print(f"\nLoading embedding model: {args.model}...")
    embedder = SentenceTransformer(args.model)

    print("Embedding texts...")
    X_train = embedder.encode([t["text"] for t in train_turns], show_progress_bar=True)
    X_test = embedder.encode([t["text"] for t in test_turns], show_progress_bar=True)
    X_cross = None
    if cross_test_turns:
        X_cross = embedder.encode([t["text"] for t in cross_test_turns], show_progress_bar=True)

    # Train and evaluate
    print(f"\n{'#'*50}")
    print(f"# {args.source.upper()} → {args.source.upper()} (same-domain)")
    print(f"{'#'*50}\n")

    results = {}
    for field in FIELDS:
        y_train = [t["label"][field] for t in train_turns]
        y_test = [t["label"][field] for t in test_turns]
        results[field] = train_and_eval(X_train, y_train, X_test, y_test, field, args.seed)

    overall = np.mean(list(results.values()))
    print(f"\n{'='*50}")
    print(f"OVERALL: {overall:.1%}")
    for f, acc in results.items():
        print(f"  {f:25s} {acc:.1%}")

    # Cross-eval
    if X_cross is not None:
        print(f"\n{'#'*50}")
        print(f"# SILVER → GOLD (cross-domain)")
        print(f"{'#'*50}\n")

        cross_results = {}
        for field in FIELDS:
            y_train = [t["label"][field] for t in train_turns]
            y_cross = [t["label"][field] for t in cross_test_turns]
            clf = LogisticRegression(max_iter=1000, random_state=args.seed, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_cross)
            acc = accuracy_score(y_cross, y_pred)
            cross_results[field] = acc
            print(f"{field}: {acc:.1%}")
            print(classification_report(y_cross, y_pred, zero_division=0))

        cross_overall = np.mean(list(cross_results.values()))
        print(f"\nCROSS-DOMAIN OVERALL: {cross_overall:.1%}")
        for f, acc in cross_results.items():
            print(f"  {f:25s} {acc:.1%}")

    # Baselines
    print(f"\n--- Baselines ---")
    print(f"  Majority class:")
    for field in FIELDS:
        y_all = [t["label"][field] for t in turns]
        majority = Counter(y_all).most_common(1)[0]
        print(f"    {field:25s} {majority[0]:10s} = {majority[1]/len(y_all):.1%}")


if __name__ == "__main__":
    main()
