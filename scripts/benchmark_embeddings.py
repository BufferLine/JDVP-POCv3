#!/usr/bin/env python3
"""Benchmark multiple embedding models for JDVP 3-axis level prediction.

Loads sharegpt-3axis silver labels (from a reference LLM like gpt4.1-cot),
embeds user turns with various sentence-transformer models, trains Ridge
regression for JH/CP/ID, derives DA, and logs all results to JSON.

Usage:
    # Run all default models against gpt4.1-cot labels
    python3 scripts/benchmark_embeddings.py

    # Single model
    python3 scripts/benchmark_embeddings.py --models all-MiniLM-L6-v2

    # Custom reference labels
    python3 scripts/benchmark_embeddings.py --ref-label sharegpt-3axis-sonnet-cot

    # Use fine-tuned model from local path
    python3 scripts/benchmark_embeddings.py --models models/jdvp-embedding-v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]

# DA derivation coefficients (from derived-da study)
DA_WEIGHTS = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570, "information_dependency": 0.268}
AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]

DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "thenlper/gte-small",
    "thenlper/gte-base",
]

BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "embedding"


def load_sharegpt_turns(ref_label: str) -> list[dict]:
    """Load turns from sharegpt 3-axis silver labels + source interactions."""
    label_dir = ROOT / "data" / "silver" / ref_label
    interaction_dir = ROOT / "data" / "open-data" / "sharegpt" / "interactions"

    if not label_dir.exists():
        print(f"ERROR: label dir not found: {label_dir}")
        sys.exit(1)
    if not interaction_dir.exists():
        print(f"ERROR: interaction dir not found: {interaction_dir}")
        sys.exit(1)

    # Load interaction texts
    interaction_texts: dict[str, dict[int, dict]] = {}
    for f in sorted(interaction_dir.glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        interaction_texts[iid] = {t["turn_number"]: t for t in d["turns"]}

    turns = []
    for conv_dir in sorted(label_dir.iterdir()):
        if not conv_dir.is_dir():
            continue
        iid = conv_dir.name
        if iid not in interaction_texts:
            continue

        conv_turns = interaction_texts[iid]
        for tf in sorted(conv_dir.glob("turn-*.json")):
            try:
                label = json.loads(tf.read_text())
            except (json.JSONDecodeError, ValueError):
                continue

            # Need all 3 axes
            if not all(a in label for a in AXES):
                continue

            tn = int(tf.stem.replace("turn-", ""))
            if tn not in conv_turns:
                continue

            human_input = conv_turns[tn].get("human_input", "")

            # Build history from prior user turns
            prior_tns = sorted([t for t in conv_turns if t < tn])
            history = [conv_turns[t]["human_input"] for t in prior_tns[-3:]]
            history_text = ""
            if history:
                history_text = " | ".join(h[-200:] for h in history) + " || "

            turn_text = history_text + f"Human: {human_input}"

            axes_vals = {a: float(label[a]) for a in AXES}
            da_derived = sum(DA_WEIGHTS[a] * axes_vals[a] for a in AXES)

            turns.append({
                "interaction_id": iid,
                "turn_number": tn,
                "text": turn_text,
                "axes": axes_vals,
                "da_derived": round(da_derived, 3),
            })

    return turns


def split_by_interaction(turns: list[dict], test_ratio: float = 0.3, seed: int = 42):
    """Split by interaction_id to prevent leakage."""
    rng = np.random.RandomState(seed)
    iids = sorted(set(t["interaction_id"] for t in turns))
    rng.shuffle(iids)
    test_count = max(1, int(len(iids) * test_ratio))
    test_iids = set(iids[:test_count])
    train = [t for t in turns if t["interaction_id"] not in test_iids]
    test = [t for t in turns if t["interaction_id"] in test_iids]
    return train, test


def bucket3(v: float) -> int:
    """Low(0-3) / Mid(4-6) / High(7-10)."""
    return 0 if v <= 3 else (1 if v <= 6 else 2)


def evaluate_model(
    model_name: str,
    train_turns: list[dict],
    test_turns: list[dict],
) -> dict:
    """Embed, train Ridge per axis, evaluate, return metrics dict."""
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    # Load and embed
    print(f"  Loading model...")
    load_start = time.time()
    try:
        trust_remote = "nomic" in model_name.lower()
        embedder = SentenceTransformer(model_name, trust_remote_code=trust_remote)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return {"model": model_name, "error": str(e)}

    load_time = time.time() - load_start
    dim = embedder.get_sentence_embedding_dimension()
    print(f"  Loaded in {load_time:.1f}s — dim={dim}")

    print(f"  Embedding train ({len(train_turns)})...")
    enc_start = time.time()
    X_train = embedder.encode([t["text"] for t in train_turns], show_progress_bar=True, batch_size=64)
    train_enc_time = time.time() - enc_start

    print(f"  Embedding test ({len(test_turns)})...")
    enc_start = time.time()
    X_test = embedder.encode([t["text"] for t in test_turns], show_progress_bar=True, batch_size=64)
    test_enc_time = time.time() - enc_start

    print(f"  Encode times: train={train_enc_time:.1f}s, test={test_enc_time:.1f}s")

    # Sanitize and normalize embeddings
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Train per-axis Ridge regressors (suppress sklearn matmul warnings on float32)
    axis_results = {}
    da_pred_all = np.zeros(len(test_turns))
    da_true_all = np.array([t["da_derived"] for t in test_turns])

    for axis in AXES:
        y_train = np.array([t["axes"][axis] for t in train_turns])
        y_test = np.array([t["axes"][axis] for t in test_turns])

        reg = Ridge(alpha=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reg.fit(X_train, y_train)
            y_pred = np.clip(reg.predict(X_test), 0, 10)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        bucket_acc = accuracy_score(
            [bucket3(v) for v in y_test],
            [bucket3(v) for v in y_pred],
        )

        axis_results[axis] = {
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "bucket_acc": round(bucket_acc, 4),
            "y_test_mean": round(float(y_test.mean()), 2),
            "y_test_std": round(float(y_test.std()), 2),
            "y_pred_mean": round(float(y_pred.mean()), 2),
            "y_pred_std": round(float(y_pred.std()), 2),
        }

        # Accumulate for derived DA
        da_pred_all += DA_WEIGHTS[axis] * y_pred

        print(f"  {axis:25s}  MAE={mae:.2f}  R²={r2:.3f}  Bucket={bucket_acc:.1%}")

    # Derived DA evaluation
    da_mae = mean_absolute_error(da_true_all, da_pred_all)
    da_r2 = r2_score(da_true_all, da_pred_all)
    da_bucket_acc = accuracy_score(
        [bucket3(v) for v in da_true_all],
        [bucket3(v) for v in da_pred_all],
    )

    # High DA detection (>= 7)
    da_high_true = [1 if v >= 7 else 0 for v in da_true_all]
    da_high_pred = [1 if v >= 7 else 0 for v in da_pred_all]
    high_tp = sum(t == 1 and p == 1 for t, p in zip(da_high_true, da_high_pred))
    high_fp = sum(t == 0 and p == 1 for t, p in zip(da_high_true, da_high_pred))
    high_fn = sum(t == 1 and p == 0 for t, p in zip(da_high_true, da_high_pred))
    high_precision = high_tp / (high_tp + high_fp) if (high_tp + high_fp) > 0 else 0
    high_recall = high_tp / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0
    high_f1 = 2 * high_precision * high_recall / (high_precision + high_recall) if (high_precision + high_recall) > 0 else 0

    da_results = {
        "mae": round(da_mae, 3),
        "r2": round(da_r2, 4),
        "bucket_acc": round(da_bucket_acc, 4),
        "high_da_precision": round(high_precision, 4),
        "high_da_recall": round(high_recall, 4),
        "high_da_f1": round(high_f1, 4),
    }

    total_time = time.time() - t0
    print(f"  DA (derived)               MAE={da_mae:.2f}  R²={da_r2:.3f}  Bucket={da_bucket_acc:.1%}")
    print(f"  High DA (>=7)              P={high_precision:.2f}  R={high_recall:.2f}  F1={high_f1:.2f}")
    print(f"  Total time: {total_time:.1f}s")

    # Baselines (predict mean)
    baseline = {}
    for axis in AXES:
        y_all = [t["axes"][axis] for t in train_turns]
        mean_val = np.mean(y_all)
        y_test = np.array([t["axes"][axis] for t in test_turns])
        baseline[axis] = round(float(np.mean(np.abs(y_test - mean_val))), 3)
    da_all = [t["da_derived"] for t in train_turns]
    da_mean = np.mean(da_all)
    baseline["da_derived"] = round(float(np.mean(np.abs(da_true_all - da_mean))), 3)

    return {
        "model": model_name,
        "embedding_dim": dim,
        "axes": axis_results,
        "da_derived": da_results,
        "baseline_mae": baseline,
        "timing": {
            "model_load_s": round(load_time, 2),
            "train_encode_s": round(train_enc_time, 2),
            "test_encode_s": round(test_enc_time, 2),
            "total_s": round(total_time, 2),
        },
    }


def print_summary(results: list[dict]) -> None:
    """Print comparison table."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nNo successful results.")
        return

    print(f"\n{'#'*70}")
    print(f"  EMBEDDING MODEL BENCHMARK SUMMARY")
    print(f"{'#'*70}\n")

    # Header
    header = f"{'Model':<35s} {'Dim':>4s} {'DA R²':>6s} {'DA MAE':>6s} {'DA Bkt':>6s} {'Hi F1':>5s} {'Time':>5s}"
    print(header)
    print("-" * len(header))

    for r in sorted(valid, key=lambda x: x["da_derived"]["r2"], reverse=True):
        name = r["model"].split("/")[-1][:34]
        print(
            f"{name:<35s} "
            f"{r['embedding_dim']:>4d} "
            f"{r['da_derived']['r2']:>6.3f} "
            f"{r['da_derived']['mae']:>6.2f} "
            f"{r['da_derived']['bucket_acc']:>6.1%} "
            f"{r['da_derived']['high_da_f1']:>5.2f} "
            f"{r['timing']['total_s']:>5.0f}s"
        )

    # Baseline
    if valid:
        bl = valid[0]["baseline_mae"]
        print(f"\n  Baseline (predict mean) DA MAE: {bl['da_derived']:.2f}")

    print(f"\n  Per-axis breakdown (best model):")
    best = max(valid, key=lambda x: x["da_derived"]["r2"])
    for axis in AXES:
        a = best["axes"][axis]
        bl_mae = best["baseline_mae"][axis]
        print(f"    {axis:25s}  R²={a['r2']:.3f}  MAE={a['mae']:.2f} (baseline={bl_mae:.2f})  Bucket={a['bucket_acc']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models for JDVP 3-axis prediction")
    parser.add_argument("--ref-label", default="sharegpt-3axis-gpt41-cot",
                        help="Silver label directory name under data/silver/")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Embedding model names (default: all defaults)")
    parser.add_argument("--include-finetuned", action="store_true",
                        help="Also test models/jdvp-embedding-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-named in data/benchmarks/embedding/)")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS[:]
    if args.include_finetuned:
        ft_path = str(ROOT / "models" / "jdvp-embedding-v1")
        if ft_path not in models:
            models.append(ft_path)

    # Load data
    print(f"Loading turns from {args.ref_label}...")
    turns = load_sharegpt_turns(args.ref_label)
    print(f"  Total turns: {len(turns)} from {len(set(t['interaction_id'] for t in turns))} conversations")

    train_turns, test_turns = split_by_interaction(turns, args.test_ratio, args.seed)
    print(f"  Train: {len(train_turns)}, Test: {len(test_turns)}")

    # Run benchmarks
    results = []
    for model_name in models:
        result = evaluate_model(model_name, train_turns, test_turns)
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.output) if args.output else BENCHMARK_DIR / f"embedding-bench-{timestamp}.json"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ref_label": args.ref_label,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "n_turns_total": len(turns),
        "n_turns_train": len(train_turns),
        "n_turns_test": len(test_turns),
        "n_conversations": len(set(t["interaction_id"] for t in turns)),
        "da_weights": DA_WEIGHTS,
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
