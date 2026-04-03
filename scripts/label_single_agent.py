"""Single-agent labeling script for use with omc-teams CLI workers.

Each CLI worker (claude, codex, gemini) runs this script to independently
label WildChat interactions with JDVP JSV fields.

Usage (called by each omc-teams worker):
    python scripts/label_single_agent.py \
        --dataset-dir data/open-data/wildchat/v1 \
        --agent-name claude \
        --max-items 50

Output:
    data/silver/wildchat-3agent-v1/agent-labels/{agent_name}/
      {interaction_id}/turn-{N}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol_core.enums import CORE_FIELD_NAMES

CORE_FIELDS = CORE_FIELD_NAMES

SYSTEM_PROMPT = """You are an expert behavioral analyst specializing in human-AI interaction dynamics. Your task: observe a single human turn (with conversation history) and characterize the human's cognitive and decisional state.

Think like a qualitative researcher coding interview transcripts — read the human's words closely, infer what they reveal about the person's mental posture, and assign codes based on your expert judgment.

# Coding Dimensions

judgment_holder — Where does decision authority sit in this turn?
  Human | Shared | AI | Undefined

delegation_awareness — Does the human show metacognitive awareness of relying on AI?
  Explicit | Implicit | Absent

cognitive_engagement — How much cognitive effort is the human investing?
  Active | Reactive | Passive

information_seeking — Is the human seeking information beyond what was provided?
  Active | Passive | None

Each dimension is independent. Code based on observable evidence in the text.

# Output

Return JSON only:
- judgment_holder, delegation_awareness, cognitive_engagement, information_seeking
- confidence: { per-field high|medium|low }
- evidence_spans: [ { text, category, note? } ]
- observer_notes: brief reasoning for your coding decisions

First character: `{`. Last character: `}`. No markdown, no commentary outside JSON."""


def build_user_prompt(
    *,
    interaction_id: str,
    turn_number: int,
    human_input: str,
    ai_response: str,
    context_turns: list[dict[str, Any]],
) -> str:
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


def validate_jsv(data: dict[str, Any]) -> bool:
    """Check that a parsed JSON has all required JSV fields with valid values."""
    valid_values = {
        "judgment_holder": {"Human", "Shared", "AI", "Undefined"},
        "delegation_awareness": {"Explicit", "Implicit", "Absent"},
        "cognitive_engagement": {"Active", "Reactive", "Passive"},
        "information_seeking": {"Active", "Passive", "None"},
    }
    for field, allowed in valid_values.items():
        if data.get(field) not in allowed:
            return False
    if not isinstance(data.get("confidence"), dict):
        return False
    if not isinstance(data.get("evidence_spans"), list) or len(data["evidence_spans"]) == 0:
        return False
    return True


def _call_llm(provider, system_prompt: str, user_prompt: str, max_attempts: int = 3) -> dict | None:
    """Call LLM and normalize response. Returns jsv dict or None."""
    from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response

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
                print(f"    LLM error: {exc}")
            time.sleep(1)
    return None


def _create_provider(base_url: str, api_key: str, model: str, timeout: float = 120.0):
    """Create an LLM provider via bufferline-llm."""
    from bufferline_llm import create_client, LlmConfig
    from src.method.tracks.llm_observer import BufferlineLlmAdapter

    config = LlmConfig(
        provider="openai",
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout,
        prefer_json_mode=True,
    )
    return BufferlineLlmAdapter(client=create_client(config), model_id=model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent JDVP labeling via LLM")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--agent-name", type=str, required=True,
                        help="Agent identifier (e.g. gemma3-12b)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. gemma3:12b)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1",
                        help="LLM API base URL (default: ollama)")
    parser.add_argument("--api-key", type=str, default="ollama",
                        help="API key (default: ollama)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_name = manifest.get("dataset_name", "unknown")

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "data" / "silver" / f"{dataset_name}-3agent-v1" / "agent-labels" / args.agent_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = manifest.get("items", [])
    if args.split:
        items = [it for it in items if it.get("split") == args.split]
    if args.max_items:
        items = items[:args.max_items]

    print(f"[{args.agent_name}] Creating provider: {args.model} @ {args.base_url}")
    provider = _create_provider(args.base_url, args.api_key, args.model, args.timeout)

    print(f"[{args.agent_name}] Labeling {len(items)} interactions from '{dataset_name}'")
    print(f"[{args.agent_name}] Output: {output_dir}")

    labeled = 0
    skipped = 0
    failed = 0
    start = time.time()

    for item_idx, item in enumerate(items):
        iid = item["interaction_id"]
        interaction_path = dataset_dir / item["relative_path"]

        if not interaction_path.exists():
            print(f"  SKIP {iid}: not found")
            continue

        interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
        turns = interaction.get("turns", [])

        iid_dir = output_dir / iid
        iid_dir.mkdir(parents=True, exist_ok=True)

        for turn in turns:
            turn_number = turn["turn_number"]
            out_path = iid_dir / f"turn-{turn_number}.json"

            if out_path.exists():
                skipped += 1
                continue

            context_turns = [t for t in turns if t["turn_number"] < turn_number]

            user_prompt = build_user_prompt(
                interaction_id=iid,
                turn_number=turn_number,
                human_input=turn["human_input"],
                ai_response=turn["ai_response"],
                context_turns=context_turns,
            )

            result = _call_llm(provider, SYSTEM_PROMPT, user_prompt)
            if result is None:
                failed += 1
                continue

            out_path.write_text(json.dumps(result.get("jsv_hint", result), indent=2, ensure_ascii=False), encoding="utf-8")
            labeled += 1

        elapsed = time.time() - start
        rate = labeled / elapsed if elapsed > 0 else 0
        print(f"  [{item_idx + 1}/{len(items)}] {iid}: {len(turns)} turns | labeled={labeled} failed={failed} | {rate:.1f} t/s")

    elapsed = time.time() - start
    print(f"\n[{args.agent_name}] Done: {labeled} labeled, {skipped} skipped, {failed} failed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
