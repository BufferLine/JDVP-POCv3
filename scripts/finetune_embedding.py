#!/usr/bin/env python3
"""Fine-tune sentence-transformer models for JDVP 3-axis DA prediction.

Generates contrastive pairs from silver labels (DA similarity), then fine-tunes
with CosineSimilarityLoss. Saves model and runs benchmark_embeddings for eval.

Usage:
    # Fine-tune bge-base (default)
    python3 scripts/finetune_embedding.py

    # Fine-tune specific model
    python3 scripts/finetune_embedding.py --base-model all-MiniLM-L6-v2

    # Fine-tune multiple models sequentially
    python3 scripts/finetune_embedding.py --base-model BAAI/bge-base-en-v1.5 BAAI/bge-small-en-v1.5 all-MiniLM-L6-v2

    # Custom epochs/batch size
    python3 scripts/finetune_embedding.py --epochs 5 --batch-size 32
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# DA derivation coefficients
DA_WEIGHTS = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570, "information_dependency": 0.268}
AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]


def load_turns(ref_labels: list[str]) -> list[dict]:
    """Load turns from one or more silver label sets, averaging DA across them."""
    interaction_dir = ROOT / "data" / "open-data" / "sharegpt" / "interactions"

    # Load interaction texts
    interaction_texts: dict[str, dict[int, dict]] = {}
    for f in sorted(interaction_dir.glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        interaction_texts[iid] = {t["turn_number"]: t for t in d["turns"]}

    # Collect labels per (iid, tn) across all ref_labels
    all_labels: dict[tuple[str, int], list[dict]] = {}
    for ref in ref_labels:
        label_dir = ROOT / "data" / "silver" / ref
        if not label_dir.exists():
            print(f"  WARNING: {label_dir} not found, skipping")
            continue
        for conv_dir in sorted(label_dir.iterdir()):
            if not conv_dir.is_dir():
                continue
            iid = conv_dir.name
            for tf in sorted(conv_dir.glob("turn-*.json")):
                try:
                    label = json.loads(tf.read_text())
                except (json.JSONDecodeError, ValueError):
                    continue
                if not all(a in label for a in AXES):
                    continue
                tn = int(tf.stem.replace("turn-", ""))
                key = (iid, tn)
                if key not in all_labels:
                    all_labels[key] = []
                all_labels[key].append({a: float(label[a]) for a in AXES})

    # Build turns with averaged labels
    turns = []
    for (iid, tn), labels in sorted(all_labels.items()):
        if iid not in interaction_texts or tn not in interaction_texts[iid]:
            continue

        # Average across label sources
        avg_axes = {}
        for a in AXES:
            avg_axes[a] = np.mean([lb[a] for lb in labels])

        da = sum(DA_WEIGHTS[a] * avg_axes[a] for a in AXES)

        conv_turns = interaction_texts[iid]
        human_input = conv_turns[tn].get("human_input", "")
        prior_tns = sorted([t for t in conv_turns if t < tn])
        history = [conv_turns[t]["human_input"] for t in prior_tns[-3:]]
        history_text = ""
        if history:
            history_text = " | ".join(h[-200:] for h in history) + " || "

        turns.append({
            "interaction_id": iid,
            "turn_number": tn,
            "text": history_text + f"Human: {human_input}",
            "da": round(da, 3),
            "n_sources": len(labels),
        })

    return turns


def generate_pairs(turns: list[dict], max_pairs: int = 15000, seed: int = 42) -> list[tuple[str, str, float]]:
    """Generate contrastive pairs with cosine similarity labels based on DA distance."""
    rng = random.Random(seed)

    # DA → similarity: same DA = 1.0, max distance (10) = 0.0
    def da_to_sim(da1: float, da2: float) -> float:
        return 1.0 - abs(da1 - da2) / 10.0

    n = len(turns)
    # Sample pairs with stratified DA distance
    pairs = []

    # All possible indices
    indices = list(range(n))

    # Generate diverse pairs
    attempts = 0
    max_attempts = max_pairs * 5
    seen = set()

    while len(pairs) < max_pairs and attempts < max_attempts:
        i, j = rng.sample(indices, 2)
        if (i, j) in seen or (j, i) in seen:
            attempts += 1
            continue
        seen.add((i, j))

        sim = da_to_sim(turns[i]["da"], turns[j]["da"])
        pairs.append((turns[i]["text"], turns[j]["text"], sim))
        attempts += 1

    # Ensure we have close and distant pairs
    # Sort by DA and add explicit close/distant pairs
    sorted_turns = sorted(enumerate(turns), key=lambda x: x[1]["da"])
    for k in range(min(1000, n - 1)):
        if len(pairs) >= max_pairs:
            break
        # Close pair (adjacent DA)
        i_idx, i_turn = sorted_turns[k]
        j_idx, j_turn = sorted_turns[k + 1]
        if (i_idx, j_idx) not in seen:
            sim = da_to_sim(i_turn["da"], j_turn["da"])
            pairs.append((i_turn["text"], j_turn["text"], sim))
            seen.add((i_idx, j_idx))

        # Distant pair (opposite end)
        far_k = n - 1 - k
        if far_k > k:
            f_idx, f_turn = sorted_turns[far_k]
            if (i_idx, f_idx) not in seen:
                sim = da_to_sim(i_turn["da"], f_turn["da"])
                pairs.append((i_turn["text"], f_turn["text"], sim))
                seen.add((i_idx, f_idx))

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def finetune_model(
    base_model: str,
    pairs: list[tuple[str, str, float]],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """Fine-tune a sentence-transformer model with contrastive pairs."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {base_model}")
    print(f"  Pairs: {len(pairs)}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"{'='*60}")

    t0 = time.time()

    trust_remote = "nomic" in base_model.lower()
    model = SentenceTransformer(base_model, trust_remote_code=trust_remote)

    # Create training examples
    examples = [InputExample(texts=[t1, t2], label=sim) for t1, t2, sim in pairs]
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.CosineSimilarityLoss(model)

    warmup_steps = int(len(dataloader) * epochs * warmup_ratio)
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Training...")

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=str(output_dir),
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s → {output_dir}")

    # Save metadata
    meta = {
        "base_model": base_model,
        "n_pairs": len(pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "training_time_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2))

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune embedding models for JDVP")
    parser.add_argument("--base-model", nargs="+",
                        default=["BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"],
                        help="Base models to fine-tune")
    parser.add_argument("--ref-labels", nargs="+",
                        default=["sharegpt-3axis-gpt41-cot", "sharegpt-3axis-sonnet-cot",
                                 "sharegpt-3axis-gemma4-26b-cot", "sharegpt-3axis-deepseek-cot"],
                        help="Silver label dirs to use (multi-model consensus)")
    parser.add_argument("--max-pairs", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", type=str, default=None,
                        help="Base dir for models (default: models/)")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip post-training benchmark")
    args = parser.parse_args()

    output_base = Path(args.output_base) if args.output_base else ROOT / "models"

    # Load data
    print(f"Loading turns from {len(args.ref_labels)} label sources...")
    turns = load_turns(args.ref_labels)
    print(f"  Total: {len(turns)} turns, {len(set(t['interaction_id'] for t in turns))} conversations")

    da_vals = [t["da"] for t in turns]
    print(f"  DA range: [{min(da_vals):.1f}, {max(da_vals):.1f}], mean={np.mean(da_vals):.2f}, std={np.std(da_vals):.2f}")

    # Generate pairs
    print(f"\nGenerating contrastive pairs (max {args.max_pairs})...")
    pairs = generate_pairs(turns, max_pairs=args.max_pairs, seed=args.seed)
    print(f"  Generated: {len(pairs)} pairs")

    sim_vals = [p[2] for p in pairs]
    print(f"  Similarity range: [{min(sim_vals):.2f}, {max(sim_vals):.2f}], mean={np.mean(sim_vals):.2f}")

    # Fine-tune each model
    trained_models = []
    for base in args.base_model:
        short_name = base.split("/")[-1]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        model_dir = output_base / f"jdvp-embedding-{short_name}-{timestamp}"

        model_path = finetune_model(
            base_model=base,
            pairs=pairs,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        trained_models.append(str(model_path))

    # Run benchmark on all trained models
    if not args.skip_benchmark and trained_models:
        print(f"\n{'#'*60}")
        print(f"  BENCHMARKING FINE-TUNED MODELS")
        print(f"{'#'*60}")

        import subprocess
        bench_models = trained_models + [str(ROOT / "models" / "jdvp-embedding-v1")]
        cmd = [
            sys.executable, str(ROOT / "scripts" / "benchmark_embeddings.py"),
            "--models", *bench_models,
        ]
        subprocess.run(cmd)

    print(f"\nTrained models:")
    for m in trained_models:
        print(f"  {m}")


if __name__ == "__main__":
    main()
