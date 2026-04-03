"""Download WildChat-1M and convert multi-turn conversations to JDVP interaction format.

Usage:
    python scripts/download_wildchat.py --max-conversations 500 --min-turns 3 --max-turns 20
    python scripts/download_wildchat.py --max-conversations 100 --lang en --split train

Output:
    data/open-data/wildchat/v1/manifest.json
    data/open-data/wildchat/v1/interactions/*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _stable_id(conversation_hash: str, index: int) -> str:
    """Deterministic interaction_id from conversation hash."""
    short = hashlib.sha256(conversation_hash.encode()).hexdigest()[:12]
    return f"wildchat-{short}-{index:04d}"


def _convert_conversation(conv: dict, index: int) -> dict | None:
    """Convert a single WildChat conversation to JDVP raw_interaction format.

    Returns None if the conversation doesn't meet quality criteria.
    """
    messages = conv.get("conversation", [])
    if not messages:
        return None

    # Build turns from alternating user/assistant messages
    turns: list[dict] = []
    turn_number = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") != "user":
            i += 1
            continue

        human_input = (msg.get("content") or "").strip()
        if not human_input:
            i += 1
            continue

        # Find next assistant response
        ai_response = ""
        if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
            ai_response = (messages[i + 1].get("content") or "").strip()
            i += 2
        else:
            i += 1
            continue

        if not ai_response:
            continue

        turns.append({
            "turn_number": turn_number,
            "human_input": human_input,
            "ai_response": ai_response,
        })
        turn_number += 1

    if len(turns) < 2:
        return None

    conversation_hash = conv.get("conversation_hash", f"unknown-{index}")
    interaction_id = _stable_id(conversation_hash, index)

    model_name = conv.get("model", "unknown")

    return {
        "interaction_id": interaction_id,
        "context_module": "general",
        "participants": {
            "human_id": f"wildchat-user-{conversation_hash[:8]}",
            "ai_id": model_name,
        },
        "turns": turns,
        "meta": {
            "source": "wildchat",
            "original_hash": conversation_hash,
            "model": model_name,
            "language": conv.get("language", "unknown"),
            "turn_count": len(turns),
        },
    }


def _build_manifest(
    items: list[dict],
    output_dir: Path,
    train_ratio: float = 0.7,
    test_ratio: float = 0.3,
) -> dict:
    """Build a dataset manifest compatible with the existing pipeline."""
    import random

    rng = random.Random(42)
    indices = list(range(len(items)))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)

    manifest_items = []
    for rank, idx in enumerate(indices):
        item = items[idx]
        iid = item["interaction_id"]
        turn_count = len(item["turns"])
        split = "train" if rank < train_end else "test"

        manifest_items.append({
            "interaction_id": iid,
            "scenario_id": "wildchat-organic",
            "scenario_title": "WildChat Organic Conversation",
            "relative_path": f"interactions/{iid}.json",
            "context_module": "general",
            "turn_count": turn_count,
            "split": split,
            "source_language": item.get("meta", {}).get("language", "unknown"),
            "source_model": item.get("meta", {}).get("model", "unknown"),
        })

    split_counts = {}
    for mi in manifest_items:
        s = mi["split"]
        split_counts[s] = split_counts.get(s, 0) + 1

    return {
        "schema_version": "pocv3-dataset-manifest-v1",
        "dataset_id": "open-data/wildchat/v1",
        "dataset_name": "wildchat",
        "dataset_version": "v1",
        "dataset_kind": "open-data",
        "context_module": "general",
        "scenario_pack_id": "wildchat-organic",
        "item_count": len(manifest_items),
        "split_counts": split_counts,
        "generation": {
            "source": "allenai/WildChat-1M",
            "license": "ODC-BY",
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
        },
        "items": manifest_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WildChat and convert to JDVP format")
    parser.add_argument("--max-conversations", type=int, default=500,
                        help="Maximum conversations to extract (default: 500)")
    parser.add_argument("--min-turns", type=int, default=3,
                        help="Minimum turns per conversation (default: 3)")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Maximum turns per conversation (default: 20)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Filter by language name (e.g. 'English', 'Korean'). None = all languages")
    parser.add_argument("--split", type=str, default="train",
                        help="HuggingFace dataset split to load (default: train)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "open-data" / "wildchat" / "v1"),
                        help="Output directory")
    parser.add_argument("--non-toxic-only", action="store_true", default=True,
                        help="Filter out toxic conversations (default: True)")
    parser.add_argument("--stream", action="store_true", default=True,
                        help="Stream dataset instead of downloading all (default: True)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    interactions_dir = output_dir / "interactions"
    interactions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WildChat-1M (split={args.split}, stream={args.stream})...")
    ds = load_dataset("allenai/WildChat-1M", split=args.split, streaming=args.stream)

    converted: list[dict] = []
    scanned = 0
    skipped_lang = 0
    skipped_toxic = 0
    skipped_turns = 0

    for conv in ds:
        scanned += 1
        if scanned % 10000 == 0:
            print(f"  scanned {scanned:,} conversations, converted {len(converted):,}...")

        # Quick pre-filter: WildChat 'turn' field = number of user-assistant pairs
        wc_turn_count = conv.get("turn", 1)
        if wc_turn_count < args.min_turns or wc_turn_count > args.max_turns:
            skipped_turns += 1
            continue

        # Language filter (WildChat uses full names: 'English', 'Korean', etc.)
        if args.lang and conv.get("language", "").lower() != args.lang.lower():
            skipped_lang += 1
            continue

        # Toxicity filter
        if args.non_toxic_only and conv.get("toxic", False):
            skipped_toxic += 1
            continue

        interaction = _convert_conversation(conv, len(converted))
        if interaction is None:
            skipped_turns += 1
            continue

        converted.append(interaction)

        if len(converted) >= args.max_conversations:
            break

    print(f"\nScan complete:")
    print(f"  scanned:       {scanned:,}")
    print(f"  converted:     {len(converted):,}")
    print(f"  skipped (lang):  {skipped_lang:,}")
    print(f"  skipped (toxic): {skipped_toxic:,}")
    print(f"  skipped (turns): {skipped_turns:,}")

    if not converted:
        print("No conversations matched filters. Exiting.")
        sys.exit(1)

    # Write interaction files
    print(f"\nWriting {len(converted)} interactions to {interactions_dir}/...")
    for interaction in converted:
        iid = interaction["interaction_id"]
        path = interactions_dir / f"{iid}.json"
        path.write_text(json.dumps(interaction, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write manifest
    manifest = _build_manifest(converted, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Manifest written to {manifest_path}")

    # Summary stats
    turn_counts = [len(c["turns"]) for c in converted]
    print(f"\nDataset summary:")
    print(f"  total interactions: {len(converted)}")
    print(f"  total turns:        {sum(turn_counts)}")
    print(f"  avg turns/conv:     {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"  min turns:          {min(turn_counts)}")
    print(f"  max turns:          {max(turn_counts)}")

    langs = {}
    for c in converted:
        lang = c.get("meta", {}).get("language", "unknown")
        langs[lang] = langs.get(lang, 0) + 1
    print(f"  languages:          {dict(sorted(langs.items(), key=lambda x: -x[1])[:10])}")


if __name__ == "__main__":
    main()
