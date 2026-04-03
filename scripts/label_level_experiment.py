"""Experiment: Level-based (0-10) JDVP labeling vs categorical.

Runs the same interactions through a level-based prompt and compares
consistency across models and delta patterns.

Usage:
    # Local gemma3
    python scripts/label_level_experiment.py \
        --dataset-dir data/open-data/wildchat/v1 \
        --agent-name gemma3-12b --model gemma3:12b --max-items 10

    # Cloud via API
    python scripts/label_level_experiment.py \
        --dataset-dir data/open-data/wildchat/v1 \
        --agent-name gemini-flash --model gemini-2.5-flash \
        --base-url https://generativelanguage.googleapis.com/v1beta/openai \
        --api-key $GEMINI_API_KEY --max-items 10
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

SYSTEM_PROMPT = """Rate the human's cognitive state on 4 dimensions (0-10). Focus on the CURRENT human message only.

# judgment_delegation (0-10) — Who holds judgment authority?
0=Human decides, 10=AI decides. Task requests (code, translate) are NOT judgment delegation.
  "I've decided to go with Rust" → 1
  "What should I choose?" → 8
  "Write me a function" → 2 (task, not judgment)
  "OK I'll do what you said" → 9

# delegation_awareness (0-10) — Does the human notice judgment is shifting?
0=Fully aware, 10=No awareness. JD high + DA high = unconscious delegation (key signal).
  "I know I'm relying on you but let me think" → 1
  "Just tell me what to do" (without noticing the shift) → 8
  "I'll use your input to decide myself" → 2

# cognitive_passivity (0-10) — How passive is the human?
0=Deep thinking, 10=Passive acceptance. Short ≠ passive, long ≠ active.
  "But wouldn't X cause Y? I think Z instead" → 1
  "OK what's next?" → 7
  "Fix line 45" → 3 (brief but shows analysis)

# information_dependency (0-10) — How much does the human NEED AI?
0=Has own knowledge, 10=Cannot form view without AI.
  "I've read the research, just checking one fact" → 1
  "Explain everything about this topic" → 8
  "What about other options? I read that..." → 2

Use prior ratings as anchor. Shift only with clear evidence. Dimensions are independent.
Return JSON only: {"judgment_delegation":<0-10>,"delegation_awareness":<0-10>,"cognitive_passivity":<0-10>,"information_dependency":<0-10>,"reasoning":"<1-2 sentences>"}
First character: `{`. Last character: `}`. No markdown."""

LEVEL_FIELDS = ["judgment_delegation", "delegation_awareness", "cognitive_passivity", "information_dependency"]


def _build_user_prompt(
    *,
    interaction_id: str,
    turn_number: int,
    human_input: str,
    context_turns: list[dict[str, Any]],
    prior_levels: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []

    recent = context_turns[-5:]
    if recent:
        parts.append("# Conversation History (for context only)")
        for ct in recent:
            t = ct.get("turn_number", "?")
            parts.append(f"[Turn {t}] Human: {ct.get('human_input', '')}")
            parts.append(f"[Turn {t}] AI: {ct.get('ai_response', '')}")
        parts.append("")

    if prior_levels:
        parts.append("# Your Ratings for the Previous Turn (use as anchor — only shift if THIS turn shows change)")
        for f in LEVEL_FIELDS:
            parts.append(f"  {f}: {prior_levels.get(f, '?')}")
        parts.append("")

    parts.append("# >>> CURRENT HUMAN MESSAGE TO ANALYZE <<<")
    parts.append(f"Turn {turn_number} | {interaction_id}")
    parts.append("")
    parts.append(f"Human says: {human_input}")
    parts.append("")
    parts.append("Rate the human's cognitive state at the moment they wrote this message. JSON only.")

    return "\n".join(parts)


def _parse_level_response(raw: str) -> dict[str, Any] | None:
    """Extract level values from LLM response."""
    # Find JSON object
    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None

                result = {}
                for f in LEVEL_FIELDS:
                    val = obj.get(f)
                    if isinstance(val, (int, float)) and 0 <= val <= 10:
                        result[f] = round(float(val), 1)
                    else:
                        return None
                result["reasoning"] = obj.get("reasoning", "")
                return result
    return None


class OllamaDirectProvider:
    """Direct ollama API call with custom sampling params."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/v1").rstrip("/")

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result["message"]["content"]


def _call_llm(provider, system_prompt: str, user_prompt: str, max_attempts: int = 3) -> dict[str, Any] | None:
    current_prompt = user_prompt
    for attempt in range(max_attempts):
        try:
            raw = provider.generate(system_prompt=system_prompt, user_prompt=current_prompt)
            result = _parse_level_response(raw)
            if result is not None:
                return result
            current_prompt = (
                f"{user_prompt}\n"
                "previous_response_was_invalid: true\n"
                "Return JSON with exactly 4 numeric fields (0-10) and reasoning.\n"
            )
        except Exception as exc:
            if attempt + 1 >= max_attempts:
                print(f"    LLM error: {exc}")
            time.sleep(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Level-based JDVP labeling experiment")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--agent-name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--api-key", type=str, default="ollama")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ollama-direct", action="store_true",
                        help="Use direct ollama API with optimal sampling params (temperature=1.0, top_p=0.95, top_k=64)")
    args = parser.parse_args()

    if args.ollama_direct:
        provider = OllamaDirectProvider(model=args.model, base_url=args.base_url)
    else:
        from bufferline_llm import create_client, LlmConfig
        from src.method.tracks.llm_observer import BufferlineLlmAdapter

        config = LlmConfig(
            provider="openai", model=args.model,
            base_url=args.base_url, api_key=args.api_key,
            timeout_seconds=args.timeout, prefer_json_mode=False,
        )
        provider = BufferlineLlmAdapter(client=create_client(config), model_id=args.model)

    dataset_dir = Path(args.dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_name = manifest.get("dataset_name", "unknown")

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "data" / "silver" / f"{dataset_name}-level-experiment" / args.agent_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = manifest.get("items", [])
    if args.max_items:
        items = items[:args.max_items]

    print(f"[{args.agent_name}] Level experiment: {len(items)} interactions, model={args.model}")

    labeled = 0
    failed = 0
    start = time.time()

    for item_idx, item in enumerate(items):
        iid = item["interaction_id"]
        interaction_path = dataset_dir / item["relative_path"]
        if not interaction_path.exists():
            continue

        interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
        turns = interaction.get("turns", [])

        iid_dir = output_dir / iid
        iid_dir.mkdir(parents=True, exist_ok=True)

        prior_levels = None

        for turn in turns:
            tn = turn["turn_number"]
            out_path = iid_dir / f"turn-{tn}.json"

            if out_path.exists():
                prior_levels = json.loads(out_path.read_text())
                continue

            context_turns = [t for t in turns if t["turn_number"] < tn]

            user_prompt = _build_user_prompt(
                interaction_id=iid,
                turn_number=tn,
                human_input=turn["human_input"],
                context_turns=context_turns,
                prior_levels=prior_levels,
            )

            result = _call_llm(provider, SYSTEM_PROMPT, user_prompt)
            if result is None:
                failed += 1
                continue

            # Compute deltas from prior
            if prior_levels:
                result["deltas"] = {
                    f: round(result[f] - prior_levels.get(f, result[f]), 1)
                    for f in LEVEL_FIELDS
                }

            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            prior_levels = result
            labeled += 1

        elapsed = time.time() - start
        rate = labeled / elapsed if elapsed > 0 else 0
        print(f"  [{item_idx + 1}/{len(items)}] {iid}: {len(turns)} turns | labeled={labeled} failed={failed} | {rate:.1f} t/s")

    elapsed = time.time() - start
    print(f"\n[{args.agent_name}] Done: {labeled} labeled, {failed} failed in {elapsed:.0f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
