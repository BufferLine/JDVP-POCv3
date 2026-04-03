"""Monitor labeling progress across all agents.

Usage:
    python scripts/label_progress.py
    python scripts/label_progress.py --watch          # auto-refresh every 5s
    python scripts/label_progress.py --watch --interval 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _count_labels(agent_dir: Path) -> dict[str, int]:
    """Count labels per interaction for an agent."""
    counts: dict[str, int] = {}
    if not agent_dir.exists():
        return counts
    for iid_dir in sorted(agent_dir.iterdir()):
        if not iid_dir.is_dir():
            continue
        turn_files = list(iid_dir.glob("turn-*.json"))
        counts[iid_dir.name] = len(turn_files)
    return counts


def _load_expected(manifest_path: Path) -> dict[str, int]:
    """Load expected turn counts from manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        item["interaction_id"]: item["turn_count"]
        for item in manifest.get("items", [])
    }


def show_progress(labels_dir: Path, manifest_path: Path) -> None:
    expected = _load_expected(manifest_path)
    total_expected = sum(expected.values())

    agents_dir = labels_dir
    if not agents_dir.exists():
        print("No labels directory found.")
        return

    agent_dirs = sorted([d for d in agents_dir.iterdir() if d.is_dir()])
    agent_names = [d.name for d in agent_dirs]

    if not agent_names:
        print("No agents found.")
        return

    # Header
    max_name = max(len(n) for n in agent_names)
    header = f"{'Agent':<{max_name}}  {'Done':>5} / {'Total':>5}  {'%':>6}  {'Bar':<22}  Status"
    print(header)
    print("-" * len(header))

    for agent_dir in agent_dirs:
        name = agent_dir.name
        counts = _count_labels(agent_dir)
        done = sum(counts.values())
        pct = done / total_expected * 100 if total_expected > 0 else 0
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)

        # Check which interactions are complete/partial/missing
        complete = sum(1 for iid, c in counts.items() if expected.get(iid, 0) > 0 and c >= expected.get(iid, 0))
        partial = sum(1 for iid, c in counts.items() if 0 < c < expected.get(iid, 999))
        total_interactions = len(expected)
        started = len(counts)

        if done >= total_expected:
            status = "✓ DONE"
        elif done == 0:
            status = "waiting..."
        else:
            status = f"{complete} complete, {partial} partial, {total_interactions - started} pending"

        print(f"{name:<{max_name}}  {done:>5} / {total_expected:>5}  {pct:>5.1f}%  {bar}  {status}")

    # Per-interaction detail
    print()
    iid_header = f"{'Interaction':<45} {'Turns':>5}"
    for name in agent_names:
        iid_header += f"  {name[:8]:>8}"
    print(iid_header)
    print("-" * len(iid_header))

    for iid in sorted(expected.keys()):
        exp = expected[iid]
        row = f"{iid:<45} {exp:>5}"
        for agent_dir in agent_dirs:
            counts = _count_labels(agent_dir)
            done = counts.get(iid, 0)
            if done >= exp:
                mark = f"{'✓':>8}"
            elif done > 0:
                mark = f"{done:>8}"
            else:
                mark = f"{'·':>8}"
            row += f"  {mark}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor labeling progress")
    parser.add_argument("--labels-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "silver" / "wildchat-3agent-v1" / "agent-labels"))
    parser.add_argument("--manifest", type=str,
                        default=str(PROJECT_ROOT / "data" / "open-data" / "wildchat" / "v1" / "manifest.json"))
    parser.add_argument("--watch", action="store_true", help="Auto-refresh")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    manifest_path = Path(args.manifest)

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                print(f"=== JDVP Labeling Progress === ({time.strftime('%H:%M:%S')})\n")
                show_progress(labels_dir, manifest_path)
                print(f"\n(refreshing every {args.interval}s, Ctrl+C to stop)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_progress(labels_dir, manifest_path)


if __name__ == "__main__":
    main()
