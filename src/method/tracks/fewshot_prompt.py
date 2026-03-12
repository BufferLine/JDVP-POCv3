"""Few-shot prompt track built on the LLM observer adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.pipeline.run_storage import read_json

from .llm_observer import LLMObserverTrack, OpenAICompatibleProvider


class FewshotPromptTrack(LLMObserverTrack):
    """LLM observer with injected few-shot examples."""

    track_id = "fewshot_prompt"

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        model_id: str,
        fewshot_pack_path: Path,
        prompt_version: str = "fewshot-prompt-v1",
    ) -> None:
        super().__init__(provider=provider, model_id=model_id, prompt_version=prompt_version)
        self.fewshot_pack_path = fewshot_pack_path
        self.fewshot_pack = read_json(fewshot_pack_path)
        self.system_prompt = self.system_prompt + "\n\nUse the few-shot examples embedded in the user prompt."

    def _fewshot_examples_block(self) -> str:
        rendered_examples: list[str] = []
        for example in self.fewshot_pack["examples"]:
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
            f"{self._fewshot_examples_block()}\n\n"
            "target_turn:\n"
            f"{base_prompt}"
        )


def create_env_backed_fewshot_track() -> FewshotPromptTrack:
    provider_kind = os.getenv("JDVP_LLM_PROVIDER", "openai_compatible")
    if provider_kind != "openai_compatible":
        raise RuntimeError(f"unsupported JDVP_LLM_PROVIDER: {provider_kind}")
    base_url = os.getenv("JDVP_LLM_BASE_URL")
    api_key = os.getenv("JDVP_LLM_API_KEY")
    model = os.getenv("JDVP_LLM_MODEL")
    fewshot_pack_path = os.getenv("JDVP_FEWSHOT_PACK_PATH")
    if not base_url or not api_key or not model:
        raise RuntimeError("JDVP_LLM_BASE_URL, JDVP_LLM_API_KEY, and JDVP_LLM_MODEL are required")
    if not fewshot_pack_path:
        raise RuntimeError("JDVP_FEWSHOT_PACK_PATH is required")
    provider = OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key)
    return FewshotPromptTrack(
        provider=provider,
        model_id=model,
        fewshot_pack_path=Path(fewshot_pack_path),
    )
