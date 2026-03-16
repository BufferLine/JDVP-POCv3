#!/usr/bin/env python3
"""Run the standard local validation flow for POCv3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fewshot_regression_suite import run_suite
from scripts.validate_contracts import main as validate_contracts_main
from src.pipeline.run_poc import run_poc
from src.pipeline.run_storage import read_json
from src.protocol_core.schema_sync import compare_schema_roots, validate_snapshot_manifest
from src.protocol_core.schema_validate import SIBLING_PROTOCOL_ROOT, VENDORED_PROTOCOL_ROOT


def run_pytest() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return {
        "command": [sys.executable, "-m", "pytest", "-q"],
        "summary": stdout_lines[-1] if stdout_lines else "",
    }


def run_assistant_docs_sync_check() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "scripts/check_assistant_docs_sync.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return {
        "command": [sys.executable, "scripts/check_assistant_docs_sync.py"],
        "summary": stdout_lines[-1] if stdout_lines else "",
    }


def run_schema_sync(*, require_upstream: bool) -> dict[str, Any]:
    if not SIBLING_PROTOCOL_ROOT.is_dir():
        if require_upstream:
            raise FileNotFoundError(f"missing upstream schema root: {SIBLING_PROTOCOL_ROOT}")
        validate_snapshot_manifest(VENDORED_PROTOCOL_ROOT)
        return {
            "status": "skipped",
            "reason": f"upstream schema root not available: {SIBLING_PROTOCOL_ROOT}",
        }

    diffs = compare_schema_roots(
        upstream_root=SIBLING_PROTOCOL_ROOT,
        vendored_root=VENDORED_PROTOCOL_ROOT,
    )
    if diffs:
        raise ValueError(
            "vendored schemas differ from upstream: "
            + ", ".join(diff.filename for diff in diffs)
        )
    validate_snapshot_manifest(VENDORED_PROTOCOL_ROOT)
    return {
        "status": "passed",
        "upstream_root": str(SIBLING_PROTOCOL_ROOT),
        "vendored_root": str(VENDORED_PROTOCOL_ROOT),
    }


def summarize_regression_results(results_path: Path) -> dict[str, Any]:
    results = read_json(results_path)
    field_rates = {
        str(field_name): float(rate)
        for field_name, rate in results.get("field_disagreement_rates", {}).items()
    }
    top_fields = [
        {"field_name": field_name, "rate": rate}
        for field_name, rate in sorted(
            field_rates.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    ]

    items = results.get("items", [])
    highest_items = [
        {
            "interaction_id": str(item["interaction_id"]),
            "scenario_id": str(item["scenario_id"]),
            "field_disagreement_rate": float(item["field_disagreement_rate"]),
        }
        for item in sorted(
            items,
            key=lambda item: (-float(item["field_disagreement_rate"]), str(item["interaction_id"])),
        )[:3]
    ]

    return {
        "results_path": str(results_path),
        "comparison_track": str(results["comparison_track"]),
        "item_count": int(results["item_count"]),
        "average_field_disagreement_rate": float(results["average_field_disagreement_rate"]),
        "max_field_disagreement_rate": float(results["max_field_disagreement_rate"]),
        "zero_disagreement_items": int(results["zero_disagreement_items"]),
        "top_disagreement_fields": top_fields,
        "highest_disagreement_items": highest_items,
    }


def run_validation_suite(
    *,
    output_root: Path,
    include_pytest: bool = True,
    include_regression: bool = True,
    require_upstream: bool = True,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": "pocv3-validation-suite-v1",
        "output_root": str(output_root),
        "steps": {},
    }

    validate_contracts_main()
    summary["steps"]["contracts"] = {"status": "passed"}

    summary["steps"]["assistant_docs_sync"] = {
        "status": "passed",
        **run_assistant_docs_sync_check(),
    }

    summary["steps"]["schema_sync"] = run_schema_sync(require_upstream=require_upstream)

    if include_pytest:
        summary["steps"]["pytest"] = {"status": "passed", **run_pytest()}
    else:
        summary["steps"]["pytest"] = {"status": "skipped"}

    fixture_path = ROOT / "data" / "fixtures" / "sample_interaction.json"
    smoke_run_dir = run_poc(
        input_path=fixture_path,
        run_id="validation-smoke",
        output_root=output_root / "runs",
        track_name="fixture_hint",
    )
    summary["steps"]["smoke_run"] = {"status": "passed", "run_dir": str(smoke_run_dir)}

    baseline_run_dir = run_poc(
        input_path=fixture_path,
        run_id="validation-baseline",
        output_root=output_root / "runs",
        track_name="heuristic_baseline",
    )
    summary["steps"]["baseline_run"] = {"status": "passed", "run_dir": str(baseline_run_dir)}

    if include_regression:
        regression_manifest_path = run_suite(
            config_path=ROOT / "config" / "eval" / "fewshot_regression_general_v1.json",
            output_root=output_root / "fewshot-regression",
            clean=True,
        )
        regression_manifest = read_json(regression_manifest_path)
        summary["steps"]["fewshot_regression"] = {
            "status": "passed",
            "manifest_path": str(regression_manifest_path),
            "summary": summarize_regression_results(Path(regression_manifest["results_path"])),
        }
    else:
        summary["steps"]["fewshot_regression"] = {"status": "skipped"}

    summary_path = output_root / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard local validation flow")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "validation" / "latest",
    )
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument(
        "--allow-missing-upstream",
        action="store_true",
        help="skip the upstream drift check when the sibling protocol repository is unavailable",
    )
    args = parser.parse_args()

    summary_path = run_validation_suite(
        output_root=args.output_root,
        include_pytest=not args.skip_pytest,
        include_regression=not args.skip_regression,
        require_upstream=not args.allow_missing_upstream,
    )
    print(f"Validation summary written: {summary_path}")


if __name__ == "__main__":
    main()
