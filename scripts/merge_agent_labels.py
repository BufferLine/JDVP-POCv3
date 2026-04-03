"""Merge 3-agent independent labels into silver dataset via majority vote.

Run after all 3 agents (claude, codex, gemini) finish labeling.

Usage:
    python scripts/merge_agent_labels.py \
        --labels-dir data/silver/wildchat-3agent-v1/agent-labels \
        --output-dir data/silver/wildchat-3agent-v1

Output:
    data/silver/wildchat-3agent-v1/
      silver-labels/
        {interaction_id}/turn-{N}.json   # majority-voted silver label
      summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol_core.enums import CORE_FIELD_NAMES

CORE_FIELDS = CORE_FIELD_NAMES


def _majority(values: list[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    winners = sorted([v for v, c in counts.items() if c == max_count])
    return winners[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge agent labels via majority vote")
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Directory containing per-agent label subdirectories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for silver labels")
    parser.add_argument("--min-agents", type=int, default=2,
                        help="Minimum agents required for a valid vote (default: 2)")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    silver_dir = output_dir / "silver-labels"
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Discover agents
    agent_dirs = sorted([d for d in labels_dir.iterdir() if d.is_dir()])
    agent_names = [d.name for d in agent_dirs]
    print(f"Found {len(agent_names)} agents: {agent_names}")

    if len(agent_names) < args.min_agents:
        print(f"ERROR: need at least {args.min_agents} agents, found {len(agent_names)}")
        sys.exit(1)

    # Collect all interaction/turn keys across agents
    all_turns: dict[str, dict[str, dict]] = {}  # {iid:turn-N: {agent: label}}

    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        for iid_dir in sorted(agent_dir.iterdir()):
            if not iid_dir.is_dir():
                continue
            iid = iid_dir.name
            for turn_file in sorted(iid_dir.glob("turn-*.json")):
                turn_key = f"{iid}:{turn_file.stem}"
                try:
                    label = json.loads(turn_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

                if turn_key not in all_turns:
                    all_turns[turn_key] = {}
                all_turns[turn_key][agent_name] = label

    print(f"Found {len(all_turns)} unique turn labels")

    # Majority vote
    merged = 0
    skipped = 0
    unanimous = 0
    field_agreement: dict[str, int] = {f: 0 for f in CORE_FIELDS}

    for turn_key, agent_labels in sorted(all_turns.items()):
        if len(agent_labels) < args.min_agents:
            skipped += 1
            continue

        iid, turn_stem = turn_key.split(":", 1)
        turn_number = int(turn_stem.replace("turn-", ""))

        # Extract JSV hints - handle both flat and nested formats
        hints = []
        for agent_name, label in agent_labels.items():
            hint = label.get("jsv_hint", label)  # support flat or nested
            hints.append(hint)

        # Vote per field
        silver_hint: dict[str, str] = {}
        agreement: dict[str, bool] = {}
        for field in CORE_FIELDS:
            values = [str(h.get(field, "")) for h in hints if h.get(field)]
            if not values:
                skipped += 1
                break
            silver_hint[field] = _majority(values)
            agree = Counter(values).most_common(1)[0][1] > len(values) / 2
            agreement[field] = agree
            if agree:
                field_agreement[field] += 1
        else:
            # Confidence vote
            confidence: dict[str, str] = {}
            for field in CORE_FIELDS:
                conf_values = []
                for h in hints:
                    c = h.get("confidence", {})
                    if isinstance(c, dict) and field in c:
                        conf_values.append(c[field])
                if conf_values:
                    confidence[field] = _majority(conf_values)
                else:
                    confidence[field] = "medium"
            silver_hint["confidence"] = confidence

            is_unanimous = all(agreement.values())
            if is_unanimous:
                unanimous += 1

            # Save silver label
            iid_silver_dir = silver_dir / iid
            iid_silver_dir.mkdir(parents=True, exist_ok=True)

            silver_label = {
                "interaction_id": iid,
                "turn_number": turn_number,
                "jsv_hint": silver_hint,
                "agreement": agreement,
                "num_agents": len(agent_labels),
                "agent_names": list(agent_labels.keys()),
                "unanimous": is_unanimous,
                "agent_labels": {
                    name: {f: label.get("jsv_hint", label).get(f) for f in CORE_FIELDS}
                    for name, label in agent_labels.items()
                },
            }
            (iid_silver_dir / f"turn-{turn_number}.json").write_text(
                json.dumps(silver_label, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            merged += 1

    # Summary
    summary = {
        "agents": agent_names,
        "total_turn_keys": len(all_turns),
        "merged": merged,
        "skipped": skipped,
        "unanimous": unanimous,
        "unanimous_rate": unanimous / merged if merged > 0 else 0,
        "field_agreement_rate": {
            f: field_agreement[f] / merged if merged > 0 else 0
            for f in CORE_FIELDS
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nMerge complete:")
    print(f"  merged:    {merged}")
    print(f"  skipped:   {skipped}")
    print(f"  unanimous: {unanimous} ({summary['unanimous_rate']:.1%})")
    print(f"  agreement: {summary['field_agreement_rate']}")
    print(f"  output:    {silver_dir}")


if __name__ == "__main__":
    main()
