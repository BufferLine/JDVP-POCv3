"""Few-shot prompt track built on the LLM observer adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.method.fewshot.selector import load_fewshot_pack, select_examples
from src.pipeline.run_storage import read_json

logger = logging.getLogger(__name__)

from .llm_observer import LLMObserverTrack, LLMProvider, create_env_backed_provider


class FewshotPromptTrack(LLMObserverTrack):
    """LLM observer with injected few-shot examples."""

    track_id = "fewshot_prompt"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model_id: str,
        fewshot_pack_path: Path,
        prompt_version: str = "fewshot-prompt-v1",
    ) -> None:
        super().__init__(provider=provider, model_id=model_id, prompt_version=prompt_version)
        self.fewshot_pack_path = fewshot_pack_path
        self.fewshot_pack = load_fewshot_pack(fewshot_pack_path)
        self.system_prompt = self.system_prompt + "\n\nUse the few-shot examples embedded in the user prompt."

    def _fewshot_examples_block(self, *, target_interaction_id: str, context_module: str) -> str:
        selected_examples = select_examples(
            pack=self.fewshot_pack,
            target_interaction_id=target_interaction_id,
            context_module=context_module,
            max_examples=len(self.fewshot_pack["examples"]),
        )
        if not selected_examples:
            logger.warning(
                "fewshot_prompt: 0 examples after filtering (interaction=%s, context=%s) — running as zero-shot",
                target_interaction_id, context_module,
            )
            self._zero_shot_fallback = True
        rendered_examples: list[str] = []
        for example in selected_examples:
            rendered_examples.append(
                json.dumps(
                    {
                        "input": {
                            "interaction_id": example["interaction_id"],
                            "turn_number": example["turn_number"],
                            "context_module": example["context_module"],
                            "human_input": example["human_input"],
                            "ai_response": example["ai_response"],
                        },
                        "output": {
                            "judgment_holder": example["jsv_hint"]["judgment_holder"],
                            "delegation_awareness": example["jsv_hint"]["delegation_awareness"],
                            "cognitive_engagement": example["jsv_hint"]["cognitive_engagement"],
                            "information_seeking": example["jsv_hint"]["information_seeking"],
                            "confidence": example["jsv_hint"]["confidence"],
                            "evidence_spans": example["evidence_spans"],
                            "observer_notes": example["observer_notes"],
                        },
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(rendered_examples)

    def _build_user_prompt(
        self,
        *,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> str:
        base_prompt = super()._build_user_prompt(
            interaction_id=interaction_id,
            turn_number=turn_number,
            human_input=human_input,
            ai_response=ai_response,
            context_turns=context_turns,
            context_module=context_module,
        )
        return (
            f"fewshot_pack_id: {self.fewshot_pack_path}\n"
            "fewshot_examples:\n"
            f"{self._fewshot_examples_block(target_interaction_id=interaction_id, context_module=context_module)}\n\n"
            "target_turn:\n"
            f"{base_prompt}"
        )


def create_env_backed_fewshot_track() -> FewshotPromptTrack:
    fewshot_pack_path = os.getenv("JDVP_FEWSHOT_PACK_PATH")
    if not fewshot_pack_path:
        raise RuntimeError("JDVP_FEWSHOT_PACK_PATH is required")
    provider, model_id = create_env_backed_provider()
    return FewshotPromptTrack(
        provider=provider,
        model_id=model_id,
        fewshot_pack_path=Path(fewshot_pack_path),
    )
