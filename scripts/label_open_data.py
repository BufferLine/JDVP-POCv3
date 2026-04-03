"""3-agent parallel labeling for open data interactions.

Runs 3 independent LLM agents (e.g. claude, codex, gemini) on each turn,
then applies majority vote to produce silver labels.

Usage:
    # Label with 3 agents configured via env-file
    python scripts/label_open_data.py \
        --dataset-dir data/open-data/wildchat/v1 \
        --agents-config config/labeling_agents.json \
        --max-items 50

    # Resume a partially completed run
    python scripts/label_open_data.py \
        --dataset-dir data/open-data/wildchat/v1 \
        --agents-config config/labeling_agents.json \
        --resume

Agent config file format (config/labeling_agents.json):
    {
      "agents": [
        {
          "name": "claude",
          "provider": "openai_compatible",
          "base_url": "https://api.anthropic.com/v1",
          "api_key_env": "ANTHROPIC_API_KEY",
          "model": "claude-sonnet-4-20250514",
          "timeout_seconds": 60
        },
        {
          "name": "gemini",
          "provider": "openai_compatible",
          "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
          "api_key_env": "GEMINI_API_KEY",
          "model": "gemini-2.5-flash",
          "timeout_seconds": 60
        },
        {
          "name": "codex",
          "provider": "openai_compatible",
          "base_url": "https://api.openai.com/v1",
          "api_key_env": "OPENAI_API_KEY",
          "model": "gpt-4.1-nano",
          "timeout_seconds": 60
        }
      ]
    }

Output:
    data/silver/wildchat-3agent-v1/
      manifest.json
      labels/
        {interaction_id}/
          turn-{N}.json          # per-turn silver label (majority vote)
          agents/
            {agent_name}.json    # individual agent labels
      summary.json               # labeling stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.method.evidence.prompt_loader import load_prompt
from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response
from src.protocol_core.enums import CORE_FIELD_NAMES

CORE_FIELDS = CORE_FIELD_NAMES

# Use v3 prompt by default (best for cloud models)
DEFAULT_PROMPT_VERSION = "llm_observer_system_v3.txt"


@dataclass
class AgentConfig:
    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0


def _load_agents_config(path: Path) -> list[AgentConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    agents = []
    for entry in raw["agents"]:
        api_key = os.environ.get(entry["api_key_env"], "")
        if not api_key:
            print(f"  WARNING: {entry['api_key_env']} not set, skipping agent '{entry['name']}'")
            continue
        agents.append(AgentConfig(
            name=entry["name"],
            provider=entry["provider"],
            base_url=entry["base_url"],
            api_key=api_key,
            model=entry["model"],
            timeout_seconds=entry.get("timeout_seconds", 60.0),
        ))
    return agents


def _create_provider(agent: AgentConfig):
    """Create an LLM provider from agent config."""
    from bufferline_llm import create_client, LlmConfig
    from src.method.tracks.llm_observer import BufferlineLlmAdapter

    config = LlmConfig(
        provider="openai",
        model=agent.model,
        base_url=agent.base_url,
        api_key=agent.api_key,
        timeout_seconds=agent.timeout_seconds,
        prefer_json_mode=True,
    )
    client = create_client(config)
    return BufferlineLlmAdapter(client=client, model_id=agent.model)


def _build_user_prompt(
    *,
    interaction_id: str,
    turn_number: int,
    human_input: str,
    ai_response: str,
    context_turns: list[dict[str, Any]],
) -> str:
    """Build the user prompt for a single turn (same format as LLMObserverTrack)."""
    parts: list[str] = []

    recent = context_turns[-5:]
    if recent:
        parts.append("# Prior Conversation")
        for ct in recent:
            t = ct.get("turn_number", "?")
            h = ct.get("human_input", "")
            a = ct.get("ai_response", "")
            parts.append(f"[Turn {t}] Human: {h}")
            parts.append(f"[Turn {t}] AI: {a}")
        parts.append("")

    parts.append("# Analyze This Turn")
    parts.append(f"Interaction: {interaction_id} | Context: general | Turn: {turn_number}")
    parts.append("")
    parts.append(f"Human: {human_input}")
    parts.append("")
    parts.append(f"AI response (for context — you are analyzing the human only): {ai_response}")
    parts.append("")
    parts.append("Respond with JSON only.")

    return "\n".join(parts)


def _label_turn_with_agent(
    provider,
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 3,
) -> dict[str, Any] | None:
    """Label a single turn with one agent. Returns normalized result or None."""
    current_prompt = user_prompt
    for attempt in range(max_attempts):
        try:
            raw = provider.generate(system_prompt=system_prompt, user_prompt=current_prompt)
            normalized = normalize_llm_response(raw)
            return normalized
        except LLMNormalizationError:
            current_prompt = (
                f"{user_prompt}\n"
                "previous_response_was_invalid_json: true\n"
                "repair_instruction: Return one JSON object only. No markdown. "
                "Start with {{ and end with }}. Include all 4 fields + confidence + evidence_spans.\n"
            )
        except Exception as exc:
            if attempt + 1 >= max_attempts:
                print(f"    agent error after {max_attempts} attempts: {exc}")
            time.sleep(1)
    return None


def _majority_vote(agent_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Majority vote across agent results to produce silver label."""
    hints = [r["jsv_hint"] for r in agent_results.values()]

    silver_hint: dict[str, Any] = {}
    agreement: dict[str, bool] = {}

    for field in CORE_FIELDS:
        values = [str(h[field]) for h in hints]
        counts = Counter(values)
        max_count = max(counts.values())
        winners = sorted([v for v, c in counts.items() if c == max_count])
        silver_hint[field] = winners[0]
        agreement[field] = max_count > len(hints) / 2

    # Majority vote on confidence
    confidence: dict[str, str] = {}
    for field in CORE_FIELDS:
        conf_values = [
            str(h.get("confidence", {}).get(field, "medium"))
            for h in hints
        ]
        counts = Counter(conf_values)
        max_count = max(counts.values())
        winners = sorted([v for v, c in counts.items() if c == max_count])
        confidence[field] = winners[0]
    silver_hint["confidence"] = confidence

    return {
        "jsv_hint": silver_hint,
        "agreement": agreement,
        "num_agents": len(agent_results),
        "agent_names": list(agent_results.keys()),
        "unanimous": all(agreement.values()),
    }


def _load_checkpoint(output_dir: Path) -> set[str]:
    """Load set of already-labeled (interaction_id, turn_number) keys."""
    done: set[str] = set()
    labels_dir = output_dir / "labels"
    if not labels_dir.exists():
        return done
    for iid_dir in labels_dir.iterdir():
        if not iid_dir.is_dir():
            continue
        for turn_file in iid_dir.glob("turn-*.json"):
            done.add(f"{iid_dir.name}:{turn_file.stem}")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="3-agent parallel labeling for open data")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to dataset directory with manifest.json")
    parser.add_argument("--agents-config", type=str, required=True,
                        help="Path to agent configuration JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/silver/{dataset_name}-3agent-v1)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max interactions to label")
    parser.add_argument("--split", type=str, default=None,
                        help="Only label items from this split (train/test)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint, skip already-labeled turns")
    parser.add_argument("--parallel-agents", action="store_true", default=True,
                        help="Run agents in parallel per turn (default: True)")
    parser.add_argument("--prompt-version", type=str, default=DEFAULT_PROMPT_VERSION,
                        help="System prompt file to use")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name = manifest.get("dataset_name", "unknown")

    # Load agents
    agents_config = _load_agents_config(Path(args.agents_config))
    if len(agents_config) < 2:
        print(f"ERROR: need at least 2 agents for majority vote, got {len(agents_config)}")
        sys.exit(1)
    print(f"Loaded {len(agents_config)} agents: {[a.name for a in agents_config]}")

    # Create providers
    providers = {}
    for agent in agents_config:
        try:
            providers[agent.name] = _create_provider(agent)
            print(f"  {agent.name}: {agent.model} @ {agent.base_url}")
        except Exception as exc:
            print(f"  WARNING: failed to create provider for {agent.name}: {exc}")

    if len(providers) < 2:
        print("ERROR: need at least 2 working providers")
        sys.exit(1)

    # System prompt
    system_prompt = load_prompt(args.prompt_version)

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "data" / "silver" / f"{dataset_name}-3agent-v1"
    )
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Resume checkpoint
    done_keys = _load_checkpoint(output_dir) if args.resume else set()
    if done_keys:
        print(f"Resuming: {len(done_keys)} turn labels already exist")

    # Filter items
    items = manifest.get("items", [])
    if args.split:
        items = [it for it in items if it.get("split") == args.split]
    if args.max_items:
        items = items[:args.max_items]

    print(f"\nLabeling {len(items)} interactions from '{dataset_name}'...")

    # Stats
    total_turns = 0
    labeled_turns = 0
    skipped_turns = 0
    failed_turns = 0
    agent_failures: dict[str, int] = {name: 0 for name in providers}
    unanimous_count = 0
    start_time = time.time()

    for item_idx, item in enumerate(items):
        iid = item["interaction_id"]
        rel_path = item["relative_path"]
        interaction_path = dataset_dir / rel_path

        if not interaction_path.exists():
            print(f"  SKIP {iid}: file not found")
            continue

        interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
        turns = interaction.get("turns", [])
        total_turns += len(turns)

        iid_labels_dir = labels_dir / iid
        iid_agents_dir = iid_labels_dir / "agents"
        iid_agents_dir.mkdir(parents=True, exist_ok=True)

        for turn in turns:
            turn_number = turn["turn_number"]
            turn_key = f"{iid}:turn-{turn_number}"

            if turn_key in done_keys:
                skipped_turns += 1
                continue

            human_input = turn["human_input"]
            ai_response = turn["ai_response"]

            # Build context from prior turns
            context_turns = [t for t in turns if t["turn_number"] < turn_number]

            user_prompt = _build_user_prompt(
                interaction_id=iid,
                turn_number=turn_number,
                human_input=human_input,
                ai_response=ai_response,
                context_turns=context_turns,
            )

            # Run agents (parallel or sequential)
            agent_results: dict[str, dict[str, Any]] = {}

            if args.parallel_agents and len(providers) > 1:
                with ThreadPoolExecutor(max_workers=len(providers)) as executor:
                    futures = {
                        executor.submit(
                            _label_turn_with_agent, prov, system_prompt, user_prompt
                        ): name
                        for name, prov in providers.items()
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                agent_results[name] = result
                            else:
                                agent_failures[name] = agent_failures.get(name, 0) + 1
                        except Exception:
                            agent_failures[name] = agent_failures.get(name, 0) + 1
            else:
                for name, prov in providers.items():
                    result = _label_turn_with_agent(prov, system_prompt, user_prompt)
                    if result is not None:
                        agent_results[name] = result
                    else:
                        agent_failures[name] = agent_failures.get(name, 0) + 1

            if len(agent_results) < 2:
                print(f"    FAIL {iid} turn {turn_number}: only {len(agent_results)} agent(s) succeeded")
                failed_turns += 1
                continue

            # Save individual agent results
            for agent_name, result in agent_results.items():
                agent_path = iid_agents_dir / f"{agent_name}-turn-{turn_number}.json"
                agent_path.write_text(json.dumps({
                    "interaction_id": iid,
                    "turn_number": turn_number,
                    "agent": agent_name,
                    **result,
                }, indent=2, ensure_ascii=False), encoding="utf-8")

            # Majority vote
            silver = _majority_vote(agent_results)

            # Save silver label
            silver_path = iid_labels_dir / f"turn-{turn_number}.json"
            silver_path.write_text(json.dumps({
                "interaction_id": iid,
                "turn_number": turn_number,
                "human_input": human_input,
                "ai_response": ai_response,
                **silver,
            }, indent=2, ensure_ascii=False), encoding="utf-8")

            labeled_turns += 1
            if silver["unanimous"]:
                unanimous_count += 1

        # Progress
        elapsed = time.time() - start_time
        rate = labeled_turns / elapsed if elapsed > 0 else 0
        print(
            f"  [{item_idx + 1}/{len(items)}] {iid}: "
            f"{len(turns)} turns | "
            f"total labeled: {labeled_turns} | "
            f"{rate:.1f} turns/sec"
        )

    # Write summary
    elapsed = time.time() - start_time
    summary = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "agents": [a.name for a in agents_config],
        "models": {a.name: a.model for a in agents_config},
        "prompt_version": args.prompt_version,
        "total_interactions": len(items),
        "total_turns": total_turns,
        "labeled_turns": labeled_turns,
        "skipped_turns": skipped_turns,
        "failed_turns": failed_turns,
        "unanimous_turns": unanimous_count,
        "unanimous_rate": unanimous_count / labeled_turns if labeled_turns > 0 else 0,
        "agent_failures": agent_failures,
        "elapsed_seconds": round(elapsed, 1),
        "turns_per_second": round(labeled_turns / elapsed, 2) if elapsed > 0 else 0,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write output manifest
    out_manifest = {
        "schema_version": "pocv3-silver-label-v1",
        "dataset_id": f"silver/{dataset_name}-3agent-v1",
        "source_dataset": str(dataset_dir),
        "agents": summary["agents"],
        "models": summary["models"],
        "labeled_turns": labeled_turns,
        "unanimous_rate": summary["unanimous_rate"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"Labeling complete!")
    print(f"  interactions:   {len(items)}")
    print(f"  turns labeled:  {labeled_turns}")
    print(f"  turns failed:   {failed_turns}")
    print(f"  turns skipped:  {skipped_turns}")
    print(f"  unanimous:      {unanimous_count} ({summary['unanimous_rate']:.1%})")
    print(f"  agent failures: {agent_failures}")
    print(f"  elapsed:        {elapsed:.0f}s ({summary['turns_per_second']} turns/sec)")
    print(f"  output:         {output_dir}")


if __name__ == "__main__":
    main()
