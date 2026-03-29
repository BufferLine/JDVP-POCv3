#!/usr/bin/env python3
"""Run a single JDVP track against the gold validation dataset with a version tag.

Each tagged run is stored independently under:
    data/validation/opus-synthetic/benchmarks/{tag}/

Runs are failure-safe: use --resume to continue after interruption.
Use --split fast for quick 10-item iteration, --split test for full 100-item evaluation.

Usage:
    # Fast iteration with local model
    JDVP_LLM_BASE_URL=http://localhost:11434/v1 \
    JDVP_LLM_API_KEY=dummy \
    JDVP_LLM_MODEL=gemma3:12b \
    python3 scripts/run_gold_benchmark.py --tag gemma3-12b-zs-v1 --track llm_observer --split fast

    # Full evaluation
    python3 scripts/run_gold_benchmark.py --tag heuristic-v1 --track heuristic_baseline --split test

    # Resume interrupted run
    python3 scripts/run_gold_benchmark.py --tag gemma3-12b-zs-v1 --track llm_observer --split test --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.poc_service import RunRequest, run_interaction

GOLD_ROOT = ROOT / "data" / "validation" / "opus-synthetic"
BENCHMARKS_DIR = GOLD_ROOT / "benchmarks"


def load_split_items(split: str) -> list[str]:
    splits = json.loads((GOLD_ROOT / "splits.json").read_text())
    if split not in splits:
        raise SystemExit(f"Unknown split '{split}'. Available: {list(splits.keys())}")
    items = splits[split]
    if not items:
        raise SystemExit(f"Split '{split}' is empty")
    return items


def run_single(
    interaction_path: Path,
    run_dir: Path,
    track_name: str,
    resume: bool,
) -> dict | None:
    iid = json.loads(interaction_path.read_text())["interaction_id"]
    run_id = f"{track_name}-{iid}"

    if resume and (run_dir / run_id / "manifest.json").exists():
        return {"interaction_id": iid, "status": "skipped_existing"}

    try:
        result = run_interaction(RunRequest(
            input_path=interaction_path,
            run_id=run_id,
            output_root=run_dir,
            track_name=track_name,
            resume=resume,
        ))
        return {
            "interaction_id": iid,
            "status": "completed",
            "run_dir": str(result.run_dir),
        }
    except Exception as exc:
        return {
            "interaction_id": iid,
            "status": "failed",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a JDVP track against the gold validation dataset with a version tag"
    )
    parser.add_argument("--tag", required=True, help="Version tag (e.g. gemma3-12b-zs-v1)")
    parser.add_argument("--track", required=True, help="Track name (llm_observer, heuristic_baseline, fewshot_prompt, etc.)")
    parser.add_argument("--split", default="fast", choices=["test", "fast"], help="Split to evaluate (default: fast)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed items")
    args = parser.parse_args()

    items = load_split_items(args.split)
    tag_dir = BENCHMARKS_DIR / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing benchmark_run.json
    meta_path = tag_dir / "benchmark_run.json"
    if meta_path.exists() and not args.resume:
        existing = json.loads(meta_path.read_text())
        if existing.get("status") == "completed":
            print(f"Tag '{args.tag}' already completed ({existing.get('completed_count')} items).")
            print("Use --resume to re-run, or choose a different --tag.")
            raise SystemExit(1)

    model_id = os.environ.get("JDVP_LLM_MODEL", "n/a")
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"Gold benchmark: tag={args.tag} track={args.track} split={args.split} items={len(items)} model={model_id}")
    print(f"Output: {tag_dir}")
    print()

    results = []
    completed = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for i, iid in enumerate(items):
        interaction_path = GOLD_ROOT / "interactions" / f"{iid}.json"
        if not interaction_path.exists():
            print(f"[{i+1}/{len(items)}] {iid} — MISSING")
            results.append({"interaction_id": iid, "status": "missing"})
            failed += 1
            continue

        t0 = time.time()
        print(f"[{i+1}/{len(items)}] {iid}...", end=" ", flush=True)
        result = run_single(interaction_path, tag_dir, args.track, args.resume)
        elapsed = time.time() - t0

        if result["status"] == "completed":
            completed += 1
            print(f"OK ({elapsed:.1f}s)")
        elif result["status"] == "skipped_existing":
            skipped += 1
            completed += 1
            print("SKIPPED (exists)")
        else:
            failed += 1
            print(f"FAILED: {result.get('error', '?')}")

        results.append(result)

    total_elapsed = time.time() - t_start

    # Write benchmark_run.json
    meta = {
        "schema_version": "gold-benchmark-run-v1",
        "tag": args.tag,
        "track_name": args.track,
        "split": args.split,
        "model_id": model_id,
        "prompt_version": os.environ.get("JDVP_PROMPT_VERSION", "default"),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(total_elapsed, 1),
        "item_count": len(items),
        "completed_count": completed,
        "failed_count": failed,
        "skipped_count": skipped,
        "status": "completed" if failed == 0 else "partial",
        "gold_dataset": "opus-synthetic",
        "gold_dataset_path": str(GOLD_ROOT),
        "env": {
            "JDVP_LLM_MODEL": os.environ.get("JDVP_LLM_MODEL"),
            "JDVP_LLM_BASE_URL": os.environ.get("JDVP_LLM_BASE_URL"),
            "JDVP_LLM_TIMEOUT_SECONDS": os.environ.get("JDVP_LLM_TIMEOUT_SECONDS"),
        },
        "items": results,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Done: {completed} completed, {failed} failed, {skipped} skipped ({total_elapsed:.0f}s)")
    print(f"Metadata: {meta_path}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
