#!/usr/bin/env python3
"""Quick 10-sample prompt benchmark for rapid iteration.

Usage:
    JDVP_LLM_BASE_URL=http://localhost:11434/v1 \
    JDVP_LLM_API_KEY=dummy \
    JDVP_LLM_MODEL=gemma3:12b \
    python3 scripts/quick_prompt_bench.py [--samples N] [--seed S]

Runs both gemma3:12b and gpt-oss:20b on the same N random samples,
then compares their outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ensemble_benchmark import compare_runs
from src.service.poc_service import RunRequest, run_interaction

DATASET_ROOT = ROOT / "data" / "generated-local-trials" / "local-turn-sim-trial100" / "v2"
INTERACTIONS_DIR = DATASET_ROOT / "interactions"


def run_sample(
    interaction_path: Path,
    model: str,
    output_dir: Path,
    base_url: str,
    api_key: str,
    timeout: str,
) -> Path | None:
    iid = json.loads(interaction_path.read_text())["interaction_id"]
    run_id = f"llm_observer-{iid}"
    env_backup = {}
    env_set = {
        "JDVP_LLM_BASE_URL": base_url,
        "JDVP_LLM_API_KEY": api_key,
        "JDVP_LLM_MODEL": model,
        "JDVP_LLM_TIMEOUT_SECONDS": timeout,
    }
    for k, v in env_set.items():
        env_backup[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        result = run_interaction(RunRequest(
            input_path=interaction_path,
            run_id=run_id,
            output_root=output_dir,
            track_name="llm_observer",
        ))
        return result.run_dir
    except Exception as exc:
        print(f"  FAILED {iid} ({model}): {exc}", file=sys.stderr)
        return None
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick prompt benchmark")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--timeout", default="120")
    parser.add_argument("--model-a", default="gemma3:12b")
    parser.add_argument("--model-b", default="gpt-oss:20b")
    args = parser.parse_args()

    interactions = sorted(INTERACTIONS_DIR.glob("*.json"))
    sample = random.Random(args.seed).sample(interactions, min(args.samples, len(interactions)))
    print(f"Running {len(sample)} samples with {args.model_a} vs {args.model_b}")

    t_start = time.time()
    rates: dict[str, list[float]] = {}
    total_dis = 0.0
    compared = 0

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, interaction_path in enumerate(sample):
            iid = json.loads(interaction_path.read_text())["interaction_id"]
            print(f"[{i+1}/{len(sample)}] {iid}...", end=" ", flush=True)
            t0 = time.time()

            dir_a = run_sample(interaction_path, args.model_a, td_path / "a", args.base_url, args.api_key, args.timeout)
            dir_b = run_sample(interaction_path, args.model_b, td_path / "b", args.base_url, args.api_key, args.timeout)

            if dir_a and dir_b:
                with tempfile.TemporaryDirectory() as bench_td:
                    s = compare_runs(run_dirs=[dir_a, dir_b], output_dir=Path(bench_td))
                    total_dis += s["field_disagreement_rate"]
                    compared += 1
                    for f, r in s["field_disagreement_rates"].items():
                        rates.setdefault(f, []).append(r)
                elapsed = time.time() - t0
                dis = s["field_disagreement_rate"]
                print(f"{dis:.0%} ({elapsed:.0f}s)")
            else:
                print("SKIP")

    total_elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Compared: {compared}/{len(sample)} ({total_elapsed:.0f}s total)")
    if compared > 0:
        print(f"Average disagreement: {total_dis/compared:.1%}")
        for f in sorted(rates):
            print(f"  {f}: {sum(rates[f])/len(rates[f]):.1%}")
        zero = sum(1 for i in range(compared) if all(rates[f][i] == 0 for f in rates))
        print(f"Zero-disagreement: {zero}/{compared}")


if __name__ == "__main__":
    main()
