#!/usr/bin/env python3
"""Compare tagged gold benchmark runs: accuracy vs gold labels + inter-model disagreement.

Usage:
    # Compare two models
    python3 scripts/compare_gold_benchmarks.py --tags heuristic-v1 gemma3-12b-zs-v1

    # Compare three models
    python3 scripts/compare_gold_benchmarks.py --tags heuristic-v1 gemma3-12b-zs-v1 gpt-oss-20b-zs-v1

    # Save report to file
    python3 scripts/compare_gold_benchmarks.py --tags heuristic-v1 gemma3-12b-zs-v1 --output report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocol_core.enums import CORE_FIELD_NAMES, CORE_ENUMS

GOLD_ROOT = ROOT / "data" / "validation" / "opus-synthetic"
BENCHMARKS_DIR = GOLD_ROOT / "benchmarks"
FIELDS = list(CORE_FIELD_NAMES)


def load_gold_labels() -> dict[str, list[dict]]:
    """Load gold labels keyed by interaction_id -> list of turn labels."""
    result = {}
    for f in sorted((GOLD_ROOT / "interactions").glob("*.json")):
        d = json.loads(f.read_text())
        iid = d["interaction_id"]
        turns = []
        for t in d["turns"]:
            meta = t.get("meta", {})
            gl = meta.get("gold_label", {})
            turns.append({
                "turn_number": t["turn_number"],
                "gold_label": gl,
                "contested": meta.get("contested", False),
                "gold_confidence": meta.get("gold_confidence", {}),
            })
        result[iid] = turns
    return result


def load_tag_extracts(tag: str) -> dict[str, list[dict]]:
    """Load extracts from a tagged benchmark run, keyed by interaction_id -> list of turn extracts."""
    tag_dir = BENCHMARKS_DIR / tag
    meta_path = tag_dir / "benchmark_run.json"
    if not meta_path.exists():
        raise SystemExit(f"Tag '{tag}' not found at {tag_dir}")

    meta = json.loads(meta_path.read_text())
    track_name = meta["track_name"]
    result = {}

    for run_dir in sorted(tag_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        iid = manifest.get("interaction_id", "")

        extracts_path = run_dir / "extracts" / track_name / "extracts.jsonl"
        if not extracts_path.exists():
            continue

        turns = []
        for line in extracts_path.read_text().strip().split("\n"):
            if line:
                turns.append(json.loads(line))
        result[iid] = turns

    return result


def compute_accuracy(
    gold: dict[str, list[dict]],
    extracts: dict[str, list[dict]],
) -> dict:
    """Compute per-field accuracy of extracts against gold labels."""
    field_correct = Counter()
    field_total = Counter()
    field_correct_noncontested = Counter()
    field_total_noncontested = Counter()
    per_item = []

    common_ids = sorted(set(gold.keys()) & set(extracts.keys()))

    for iid in common_ids:
        gold_turns = {t["turn_number"]: t for t in gold[iid]}
        extract_turns = {t.get("turn_number", t.get("turn")): t for t in extracts[iid]}

        item_correct = 0
        item_total = 0
        item_details = []

        for tn in sorted(set(gold_turns.keys()) & set(extract_turns.keys())):
            gt = gold_turns[tn]
            et = extract_turns[tn]
            gl = gt["gold_label"]
            contested = gt["contested"]
            jsv = et.get("jsv_hint", {})

            turn_detail = {"turn_number": tn, "contested": contested, "fields": {}}

            for field in FIELDS:
                gold_val = gl.get(field)
                pred_val = jsv.get(field)
                if gold_val is None:
                    continue

                match = gold_val == pred_val
                field_total[field] += 1
                item_total += 1
                if match:
                    field_correct[field] += 1
                    item_correct += 1
                if not contested:
                    field_total_noncontested[field] += 1
                    if match:
                        field_correct_noncontested[field] += 1

                turn_detail["fields"][field] = {
                    "gold": gold_val,
                    "predicted": pred_val,
                    "correct": match,
                }

            item_details.append(turn_detail)

        item_acc = item_correct / item_total if item_total > 0 else 0.0
        per_item.append({
            "interaction_id": iid,
            "accuracy": round(item_acc, 4),
            "correct": item_correct,
            "total": item_total,
            "turns": item_details,
        })

    # Aggregate
    field_accuracy = {}
    for field in FIELDS:
        total = field_total[field]
        correct = field_correct[field]
        field_accuracy[field] = {
            "accuracy": round(correct / total, 4) if total > 0 else None,
            "correct": correct,
            "total": total,
        }

    field_accuracy_noncontested = {}
    for field in FIELDS:
        total = field_total_noncontested[field]
        correct = field_correct_noncontested[field]
        field_accuracy_noncontested[field] = {
            "accuracy": round(correct / total, 4) if total > 0 else None,
            "correct": correct,
            "total": total,
        }

    overall_correct = sum(field_correct.values())
    overall_total = sum(field_total.values())

    return {
        "items_evaluated": len(common_ids),
        "overall_accuracy": round(overall_correct / overall_total, 4) if overall_total > 0 else None,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "field_accuracy": field_accuracy,
        "field_accuracy_noncontested": field_accuracy_noncontested,
        "per_item": per_item,
    }


def compute_disagreement(
    extracts_a: dict[str, list[dict]],
    extracts_b: dict[str, list[dict]],
) -> dict:
    """Compute pairwise field disagreement between two sets of extracts."""
    field_disagree = Counter()
    field_total = Counter()
    common_ids = sorted(set(extracts_a.keys()) & set(extracts_b.keys()))

    for iid in common_ids:
        turns_a = {t.get("turn_number", t.get("turn")): t for t in extracts_a[iid]}
        turns_b = {t.get("turn_number", t.get("turn")): t for t in extracts_b[iid]}

        for tn in sorted(set(turns_a.keys()) & set(turns_b.keys())):
            jsv_a = turns_a[tn].get("jsv_hint", {})
            jsv_b = turns_b[tn].get("jsv_hint", {})
            for field in FIELDS:
                val_a = jsv_a.get(field)
                val_b = jsv_b.get(field)
                if val_a is not None and val_b is not None:
                    field_total[field] += 1
                    if val_a != val_b:
                        field_disagree[field] += 1

    field_rates = {}
    for field in FIELDS:
        total = field_total[field]
        dis = field_disagree[field]
        field_rates[field] = round(dis / total, 4) if total > 0 else None

    overall_dis = sum(field_disagree.values())
    overall_total = sum(field_total.values())

    return {
        "items_compared": len(common_ids),
        "overall_disagreement": round(overall_dis / overall_total, 4) if overall_total > 0 else None,
        "field_disagreement_rates": field_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tagged gold benchmark runs")
    parser.add_argument("--tags", nargs="+", required=True, help="Tags to compare (2+)")
    parser.add_argument("--output", type=Path, default=None, help="Save full report to JSON file")
    args = parser.parse_args()

    if len(args.tags) < 1:
        raise SystemExit("Need at least 1 tag")

    # Load gold labels
    gold = load_gold_labels()

    # Load all tag extracts
    tag_extracts = {}
    tag_meta = {}
    for tag in args.tags:
        tag_extracts[tag] = load_tag_extracts(tag)
        meta_path = BENCHMARKS_DIR / tag / "benchmark_run.json"
        tag_meta[tag] = json.loads(meta_path.read_text())

    # === Gold Accuracy per tag ===
    print("=" * 60)
    print("GOLD ACCURACY (vs gold labels)")
    print("=" * 60)

    accuracy_results = {}
    for tag in args.tags:
        acc = compute_accuracy(gold, tag_extracts[tag])
        accuracy_results[tag] = acc
        model = tag_meta[tag].get("model_id", "?")
        track = tag_meta[tag].get("track_name", "?")
        print(f"\n  {tag} (model={model}, track={track})")
        print(f"    Overall: {acc['overall_accuracy']:.1%} ({acc['overall_correct']}/{acc['overall_total']})")
        print(f"    Items: {acc['items_evaluated']}")
        for field in FIELDS:
            fa = acc["field_accuracy"][field]
            print(f"    {field:25s} {fa['accuracy']:.1%} ({fa['correct']}/{fa['total']})")

    # === Pairwise Disagreement ===
    if len(args.tags) >= 2:
        print(f"\n{'=' * 60}")
        print("INTER-MODEL DISAGREEMENT")
        print("=" * 60)

        disagreement_results = {}
        for i, tag_a in enumerate(args.tags):
            for tag_b in args.tags[i + 1:]:
                dis = compute_disagreement(tag_extracts[tag_a], tag_extracts[tag_b])
                key = f"{tag_a} vs {tag_b}"
                disagreement_results[key] = dis
                print(f"\n  {key}")
                print(f"    Overall: {dis['overall_disagreement']:.1%}")
                print(f"    Items: {dis['items_compared']}")
                for field in FIELDS:
                    rate = dis["field_disagreement_rates"][field]
                    print(f"    {field:25s} {rate:.1%}")
    else:
        disagreement_results = {}

    # === Summary Table ===
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Field':25s}"
    for tag in args.tags:
        header += f" {tag:>16s}"
    print(header)
    print("-" * len(header))
    for field in FIELDS:
        row = f"{field:25s}"
        for tag in args.tags:
            fa = accuracy_results[tag]["field_accuracy"][field]
            row += f" {fa['accuracy']:>15.1%}"
        print(row)
    row = f"{'OVERALL':25s}"
    for tag in args.tags:
        row += f" {accuracy_results[tag]['overall_accuracy']:>15.1%}"
    print(row)

    # === Save report ===
    report = {
        "schema_version": "gold-benchmark-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": args.tags,
        "tag_metadata": {tag: {
            "model_id": tag_meta[tag].get("model_id"),
            "track_name": tag_meta[tag].get("track_name"),
            "split": tag_meta[tag].get("split"),
            "item_count": tag_meta[tag].get("item_count"),
            "completed_count": tag_meta[tag].get("completed_count"),
        } for tag in args.tags},
        "accuracy": {tag: {
            "overall_accuracy": accuracy_results[tag]["overall_accuracy"],
            "field_accuracy": {f: accuracy_results[tag]["field_accuracy"][f]["accuracy"] for f in FIELDS},
            "field_accuracy_noncontested": {f: accuracy_results[tag]["field_accuracy_noncontested"][f]["accuracy"] for f in FIELDS},
            "items_evaluated": accuracy_results[tag]["items_evaluated"],
        } for tag in args.tags},
        "disagreement": disagreement_results,
    }

    # Always save to benchmarks dir
    comparison_dir = BENCHMARKS_DIR / "_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    tag_slug = "_vs_".join(args.tags)
    auto_path = comparison_dir / f"{tag_slug}.json"
    auto_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nReport saved: {auto_path}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
