#!/usr/bin/env python3
"""Benchmark embedding models on conversation-level trend prediction.

Evaluates both score-based trends (linear slope) and bucket-based trends
(Low/Mid/High transitions) across all 4 JDVP axes (JH, CP, ID, DA).

Usage:
    python3 scripts/benchmark_embedding_trends.py
    python3 scripts/benchmark_embedding_trends.py --models models/jdvp-embedding-v1 all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]

DA_WEIGHTS = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570, "information_dependency": 0.268}
AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]
ALL_FIELDS = AXES + ["da_derived"]

DEFAULT_REF_LABELS = [
    "sharegpt-3axis-gpt41-cot",
    "sharegpt-3axis-sonnet-cot",
    "sharegpt-3axis-gemma4-26b-cot",
    "sharegpt-3axis-deepseek-cot",
]

BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "embedding"


# ---------- Trend classification ----------

def bucket3(v: float) -> int:
    """Low=0 (0-3), Mid=1 (4-6), High=2 (7-10)."""
    return 0 if v <= 3 else (1 if v <= 6 else 2)

BUCKET_NAMES = {0: "Low", 1: "Mid", 2: "High"}


def compute_slope(seq: list[float]) -> float:
    if len(seq) < 2:
        return 0.0
    x = np.arange(len(seq), dtype=float)
    x_norm = x / max(x.max(), 1)
    return float(np.polyfit(x_norm, seq, 1)[0])


def score_trend(seq: list[float], threshold: float = 0.3) -> str:
    """Trend from raw 0-10 slope."""
    slope = compute_slope(seq)
    if slope > threshold:
        return "rising"
    elif slope < -threshold:
        return "falling"
    return "stable"


def bucket_trend(seq: list[float]) -> str:
    """Trend from bucket transitions (first half avg vs second half avg)."""
    if len(seq) < 2:
        return "stable"
    mid = len(seq) // 2
    first_bucket = bucket3(np.mean(seq[:mid]))
    second_bucket = bucket3(np.mean(seq[mid:]))
    if second_bucket > first_bucket:
        return "rising"
    elif second_bucket < first_bucket:
        return "falling"
    return "stable"


def classify_conversation(axis_sequences: dict[str, list[float]]) -> dict:
    """Classify trends for all fields using both methods.

    Returns {
        "score": {field: trend, ..., "overall": trend},
        "bucket": {field: trend, ..., "overall": trend},
        "slopes": {field: float},
    }
    """
    score_trends = {}
    bucket_trends = {}
    slopes = {}

    for field, seq in axis_sequences.items():
        slopes[field] = compute_slope(seq)
        score_trends[field] = score_trend(seq)
        bucket_trends[field] = bucket_trend(seq)

    # Overall = majority vote across 4 fields
    score_vote = Counter(score_trends.values()).most_common(1)[0][0]
    bucket_vote = Counter(bucket_trends.values()).most_common(1)[0][0]
    score_trends["overall"] = score_vote
    bucket_trends["overall"] = bucket_vote

    return {"score": score_trends, "bucket": bucket_trends, "slopes": slopes}


# ---------- Data loading ----------

def load_llm_trends(ref_labels: list[str]) -> dict[str, dict]:
    """Load per-conversation trends from LLM labels (all 4 axes, both methods)."""
    # First pass: collect per-turn values per (ref, iid)
    raw: dict[str, dict[str, list[tuple[int, dict]]]] = {}  # ref -> iid -> [(tn, {field: val})]

    for ref in ref_labels:
        label_dir = ROOT / "data" / "silver" / ref
        if not label_dir.exists():
            print(f"  WARNING: {label_dir} not found")
            continue
        raw[ref] = {}

        for conv_dir in sorted(label_dir.iterdir()):
            if not conv_dir.is_dir():
                continue
            iid = conv_dir.name
            turns = []
            for tf in sorted(conv_dir.glob("turn-*.json")):
                try:
                    label = json.loads(tf.read_text())
                except (json.JSONDecodeError, ValueError):
                    continue
                if not all(a in label for a in AXES):
                    continue
                tn = int(tf.stem.replace("turn-", ""))
                vals = {a: float(label[a]) for a in AXES}
                vals["da_derived"] = sum(DA_WEIGHTS[a] * vals[a] for a in AXES)
                turns.append((tn, vals))

            if len(turns) >= 2:
                turns.sort(key=lambda x: x[0])
                raw[ref][iid] = turns

    # All conversation IDs that appear in at least 2 models
    all_iids = set()
    for ref_data in raw.values():
        all_iids |= set(ref_data.keys())

    results = {}
    for iid in sorted(all_iids):
        per_model = {}
        for ref in ref_labels:
            if ref not in raw or iid not in raw[ref]:
                continue
            turns = raw[ref][iid]
            axis_seqs = {f: [t[1][f] for t in turns] for f in ALL_FIELDS}
            per_model[ref] = classify_conversation(axis_seqs)

        if len(per_model) < 2:
            continue

        # Consensus per method per field
        consensus = {"score": {}, "bucket": {}}
        for method in ["score", "bucket"]:
            for field in ALL_FIELDS + ["overall"]:
                votes = [m[method][field] for m in per_model.values()]
                vote = Counter(votes).most_common(1)[0]
                consensus[method][field] = vote[0]

        # Compute average sequences for reference
        refs_with_data = [ref for ref in ref_labels if ref in per_model]
        min_lens = {}
        for f in ALL_FIELDS:
            lens = [len(raw[ref][iid]) for ref in refs_with_data if ref in raw and iid in raw[ref]]
            min_lens[f] = min(lens) if lens else 0
        avg_seqs = {}
        for f in ALL_FIELDS:
            ml = min_lens[f]
            if ml > 0:
                avg_seqs[f] = [
                    float(np.mean([raw[ref][iid][i][1][f] for ref in refs_with_data if ref in raw and iid in raw[ref]]))
                    for i in range(ml)
                ]
            else:
                avg_seqs[f] = []

        n_models = len(per_model)
        # Count unanimous for overall score trend
        overall_votes = [m["score"]["overall"] for m in per_model.values()]
        unanimous_count = Counter(overall_votes).most_common(1)[0][1]

        results[iid] = {
            "per_model": per_model,
            "consensus": consensus,
            "avg_sequences": avg_seqs,
            "n_models": n_models,
            "consensus_count": unanimous_count,
        }

    return results


def load_conversation_turns(ref_label: str) -> dict[str, list[dict]]:
    """Load turns with per-axis ground truth values."""
    label_dir = ROOT / "data" / "silver" / ref_label
    interaction_dir = ROOT / "data" / "open-data" / "sharegpt" / "interactions"

    interaction_texts: dict[str, dict[int, dict]] = {}
    for f in sorted(interaction_dir.glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        interaction_texts[iid] = {t["turn_number"]: t for t in d["turns"]}

    conversations: dict[str, list[dict]] = {}
    for conv_dir in sorted(label_dir.iterdir()):
        if not conv_dir.is_dir():
            continue
        iid = conv_dir.name
        if iid not in interaction_texts:
            continue

        conv_turns = interaction_texts[iid]
        turns = []
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
            prior_tns = sorted([t for t in conv_turns if t < tn])
            history = [conv_turns[t]["human_input"] for t in prior_tns[-3:]]
            history_text = ""
            if history:
                history_text = " | ".join(h[-200:] for h in history) + " || "

            axis_vals = {a: float(label[a]) for a in AXES}
            axis_vals["da_derived"] = sum(DA_WEIGHTS[a] * axis_vals[a] for a in AXES)

            turns.append({
                "interaction_id": iid,
                "turn_number": tn,
                "text": history_text + f"Human: {human_input}",
                "axes": axis_vals,
            })

        if len(turns) >= 2:
            turns.sort(key=lambda t: t["turn_number"])
            conversations[iid] = turns

    return conversations


# ---------- Evaluation ----------

def evaluate_model(
    model_name: str,
    conversations: dict[str, list[dict]],
    llm_trends: dict[str, dict],
    seed: int = 42,
) -> dict:
    """Train per-axis Ridge, predict trends, compare against LLM consensus."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    trust_remote = "nomic" in model_name.lower()
    try:
        embedder = SentenceTransformer(model_name, trust_remote_code=trust_remote)
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"model": model_name, "error": str(e)}

    dim = embedder.get_sentence_embedding_dimension()

    # Split by conversation
    common_iids = sorted(set(conversations.keys()) & set(llm_trends.keys()))
    rng = np.random.RandomState(seed)
    rng.shuffle(common_iids)
    test_count = max(1, int(len(common_iids) * 0.3))
    test_iids = set(common_iids[:test_count])
    train_iids = set(common_iids[test_count:])

    train_turns = [t for iid in train_iids for t in conversations[iid]]
    test_turns = [t for iid in test_iids for t in conversations[iid]]

    print(f"  Conv: {len(common_iids)} (train={len(train_iids)}, test={len(test_iids)})")
    print(f"  Turns: train={len(train_turns)}, test={len(test_turns)}")

    # Embed all
    all_turns = train_turns + test_turns
    print(f"  Embedding {len(all_turns)} turns...")
    X_all = embedder.encode([t["text"] for t in all_turns], show_progress_bar=True, batch_size=64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = normalize(X_all)
    X_train = X_all[:len(train_turns)]
    X_test = X_all[len(train_turns):]

    # Train per-axis Ridge regressors
    regressors = {}
    for field in ALL_FIELDS:
        y_train = np.array([t["axes"][field] for t in train_turns])
        reg = Ridge(alpha=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reg.fit(X_train, y_train)
        regressors[field] = reg

    # Predict per-axis for test turns
    predictions = {}
    for field, reg in regressors.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions[field] = np.clip(reg.predict(X_test), 0, 10)

    # Group predictions back into conversations
    idx = 0
    conv_pred_seqs: dict[str, dict[str, list[float]]] = {}
    for iid in test_iids:
        n = len(conversations[iid])
        conv_pred_seqs[iid] = {
            field: list(predictions[field][idx:idx + n]) for field in ALL_FIELDS
        }
        idx += n

    # Classify embedding trends
    emb_classified = {}
    for iid, seqs in conv_pred_seqs.items():
        emb_classified[iid] = classify_conversation(seqs)

    # Compare against LLM consensus
    results_by_method = {}
    for method in ["score", "bucket"]:
        field_results = {}
        for field in ALL_FIELDS + ["overall"]:
            y_true, y_pred = [], []
            for iid in sorted(test_iids):
                if iid not in llm_trends:
                    continue
                y_true.append(llm_trends[iid]["consensus"][method][field])
                y_pred.append(emb_classified[iid][method][field])

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=["rising", "stable", "falling"])
            field_results[field] = {
                "accuracy": round(acc, 4),
                "confusion_matrix": cm.tolist(),
                "distribution_true": dict(Counter(y_true)),
                "distribution_pred": dict(Counter(y_pred)),
            }
        results_by_method[method] = field_results

    # High-consensus accuracy (overall score trend, 3+ agree)
    hc_true, hc_pred = [], []
    for iid in sorted(test_iids):
        if iid not in llm_trends:
            continue
        if llm_trends[iid]["consensus_count"] >= 3:
            hc_true.append(llm_trends[iid]["consensus"]["score"]["overall"])
            hc_pred.append(emb_classified[iid]["score"]["overall"])
    hc_acc = accuracy_score(hc_true, hc_pred) if hc_true else None

    elapsed = time.time() - t0

    # Print results
    print(f"\n  {'SCORE TREND':^50s}")
    print(f"  {'Field':<25s} {'Accuracy':>8s}  {'Distribution (true → pred)'}")
    print(f"  {'-'*70}")
    for field in ALL_FIELDS + ["overall"]:
        r = results_by_method["score"][field]
        dt = r["distribution_true"]
        dp = r["distribution_pred"]
        dist_str = f"R:{dt.get('rising',0)} S:{dt.get('stable',0)} F:{dt.get('falling',0)} → R:{dp.get('rising',0)} S:{dp.get('stable',0)} F:{dp.get('falling',0)}"
        marker = " ◀" if field == "overall" else ""
        print(f"  {field:<25s} {r['accuracy']:>7.1%}   {dist_str}{marker}")

    print(f"\n  {'BUCKET TREND':^50s}")
    print(f"  {'Field':<25s} {'Accuracy':>8s}  {'Distribution (true → pred)'}")
    print(f"  {'-'*70}")
    for field in ALL_FIELDS + ["overall"]:
        r = results_by_method["bucket"][field]
        dt = r["distribution_true"]
        dp = r["distribution_pred"]
        dist_str = f"R:{dt.get('rising',0)} S:{dt.get('stable',0)} F:{dt.get('falling',0)} → R:{dp.get('rising',0)} S:{dp.get('stable',0)} F:{dp.get('falling',0)}"
        marker = " ◀" if field == "overall" else ""
        print(f"  {field:<25s} {r['accuracy']:>7.1%}   {dist_str}{marker}")

    if hc_acc is not None:
        print(f"\n  High-consensus (3+ agree) overall score acc: {hc_acc:.1%} ({len(hc_true)} convs)")

    print(f"\n  Total time: {elapsed:.1f}s")

    return {
        "model": model_name,
        "embedding_dim": dim,
        "score_trends": results_by_method["score"],
        "bucket_trends": results_by_method["bucket"],
        "overall_score_accuracy": results_by_method["score"]["overall"]["accuracy"],
        "overall_bucket_accuracy": results_by_method["bucket"]["overall"]["accuracy"],
        "high_consensus_accuracy": round(hc_acc, 4) if hc_acc else None,
        "n_test_conversations": len(test_iids),
        "n_test_turns": len(test_turns),
        "timing_s": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding trend prediction")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--ref-labels", nargs="+", default=DEFAULT_REF_LABELS)
    parser.add_argument("--ref-label-for-turns", default="sharegpt-3axis-gpt41-cot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    models = args.models or [str(ROOT / "models" / "jdvp-embedding-v1")]

    # Load LLM ground truth
    print("Loading LLM trends from", len(args.ref_labels), "models...")
    llm_trends = load_llm_trends(args.ref_labels)
    print(f"  {len(llm_trends)} conversations")

    # Distribution summary
    for method in ["score", "bucket"]:
        counts = Counter(d["consensus"][method]["overall"] for d in llm_trends.values())
        print(f"  {method} overall consensus: {dict(counts)}")

    unanimous = sum(1 for d in llm_trends.values() if d["consensus_count"] == d["n_models"])
    print(f"  Unanimous (score overall): {unanimous}/{len(llm_trends)}")

    # Load turns
    print(f"\nLoading turns from {args.ref_label_for_turns}...")
    conversations = load_conversation_turns(args.ref_label_for_turns)
    print(f"  {len(conversations)} conversations, {sum(len(t) for t in conversations.values())} turns")

    # Benchmark
    results = []
    for model_name in models:
        results.append(evaluate_model(model_name, conversations, llm_trends, args.seed))

    # Summary table
    valid = [r for r in results if "error" not in r]
    if valid:
        print(f"\n{'#'*70}")
        print(f"  TREND PREDICTION SUMMARY")
        print(f"{'#'*70}\n")
        header = f"{'Model':<35s} {'Score':>6s} {'Bucket':>7s} {'HC':>6s} {'Time':>5s}"
        print(header)
        print("-" * len(header))
        for r in sorted(valid, key=lambda x: x["overall_score_accuracy"], reverse=True):
            name = r["model"].split("/")[-1][:34]
            hc = f"{r['high_consensus_accuracy']:.1%}" if r.get("high_consensus_accuracy") else "N/A"
            print(f"{name:<35s} {r['overall_score_accuracy']:>6.1%} {r['overall_bucket_accuracy']:>7.1%} {hc:>6s} {r['timing_s']:>5.0f}s")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.output) if args.output else BENCHMARK_DIR / f"trend-bench-{timestamp}.json"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ref_labels": args.ref_labels,
        "seed": args.seed,
        "n_conversations": len(llm_trends),
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
