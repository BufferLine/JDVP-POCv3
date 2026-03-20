#!/usr/bin/env python3
"""Run the standard few-shot regression benchmark suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.build_fewshot_pack import build_fewshot_pack
from src.dataset.generate_dataset import generate_dataset
from src.eval.fewshot_benchmark import build_fewshot_benchmark_plan
from src.eval.run_fewshot_benchmark import run_fewshot_benchmark_plan
from src.pipeline.run_poc import run_poc
from src.pipeline.run_storage import read_json, write_json


DEFAULT_CONFIG_PATH = ROOT / "config" / "eval" / "fewshot_regression_general_v1.json"


@contextmanager
def temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def run_suite(*, config_path: Path, output_root: Path, clean: bool = False) -> Path:
    config = load_json(config_path)
    if clean and output_root.exists():
        resolved = output_root.resolve()
        if resolved == ROOT or ROOT.is_relative_to(resolved):
            raise ValueError(f"refusing to --clean a parent of the project root: {resolved}")
        shutil.rmtree(output_root)

    dataset_cfg = config["dataset"]
    pack_cfg = config["fewshot_pack"]
    benchmark_cfg = config["benchmark"]
    threshold_cfg = benchmark_cfg.get("thresholds", {})

    dataset_root = generate_dataset(
        dataset_name=str(dataset_cfg["dataset_name"]),
        dataset_version=str(dataset_cfg["dataset_version"]),
        output_root=output_root / "generated",
        scenario_pack_path=resolve_path(str(dataset_cfg["scenario_pack_path"])),
        count_per_scenario=int(dataset_cfg["count_per_scenario"]),
        seed=int(dataset_cfg["seed"]),
    )

    seed_run_dir = run_poc(
        input_path=resolve_path(str(pack_cfg["fixture_input_path"])),
        run_id=str(pack_cfg["seed_run_id"]),
        output_root=output_root / "seed-runs",
        track_name=str(pack_cfg["seed_track_name"]),
    )

    fewshot_pack_path = output_root / "fewshot-pack.json"
    build_fewshot_pack(
        run_dir=seed_run_dir,
        output_path=fewshot_pack_path,
        max_examples=int(pack_cfg["max_examples"]),
    )

    plan_path = output_root / "fewshot-plan.json"
    build_fewshot_benchmark_plan(
        dataset_root=dataset_root,
        fewshot_pack_path=fewshot_pack_path,
        split=str(benchmark_cfg["split"]),
        output_path=plan_path,
        max_examples=int(pack_cfg["max_examples"]),
    )

    provider_updates = {
        "JDVP_LLM_PROVIDER": str(benchmark_cfg["provider_kind"]),
        "JDVP_LLM_MODEL": str(benchmark_cfg.get("model_id", "static-response")),
    }
    static_response_path = benchmark_cfg.get("static_response_path")
    if static_response_path:
        provider_updates["JDVP_LLM_STATIC_RESPONSE_PATH"] = str(resolve_path(str(static_response_path)))

    with temporary_env(provider_updates):
        results_path = run_fewshot_benchmark_plan(
            plan_path=plan_path,
            output_root=output_root / "benchmark-results",
            comparison_track=str(benchmark_cfg["comparison_track"]),
            max_average_field_disagreement_rate=threshold_cfg.get("max_average_field_disagreement_rate"),
            max_field_disagreement_rate=threshold_cfg.get("max_field_disagreement_rate"),
        )

    results = read_json(results_path)
    manifest = {
        "schema_version": "pocv3-fewshot-regression-suite-v1",
        "config_path": str(config_path),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "seed_run_dir": str(seed_run_dir),
        "fewshot_pack_path": str(fewshot_pack_path),
        "plan_path": str(plan_path),
        "results_path": str(results_path),
        "summary": {
            "split": results["split"],
            "item_count": results["item_count"],
            "comparison_track": results["comparison_track"],
            "average_field_disagreement_rate": results["average_field_disagreement_rate"],
            "max_field_disagreement_rate": results["max_field_disagreement_rate"],
            "zero_disagreement_items": results["zero_disagreement_items"],
        },
    }
    manifest_path = output_root / "suite_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard few-shot regression suite")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    manifest_path = run_suite(
        config_path=args.config,
        output_root=args.output_root,
        clean=args.clean,
    )
    print(f"Few-shot regression suite written: {manifest_path}")


if __name__ == "__main__":
    main()
