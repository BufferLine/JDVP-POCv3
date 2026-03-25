"""Provider-backed LLM observer track."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from bufferline_llm import create_client, LlmConfig
from bufferline_llm.providers.openai_compat import OpenAICompatClient
from bufferline_llm.providers.static import StaticResponseClient

from src.method.evidence.prompt_loader import load_prompt
from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response

from .base import TrackExtractor, TrackOutput


class LLMProvider(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class BufferlineLlmAdapter:
    """Adapts a bufferline-llm client to the POCv3 LLMProvider protocol."""

    def __init__(self, client: OpenAICompatClient | StaticResponseClient | Any, model_id: str) -> None:
        self._client = client
        self.model_id = model_id

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        result = self._client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if not result.ok or result.content is None:
            raise RuntimeError(result.error_message or "LLM call failed")
        return result.content


class StaticResponseProvider:
    """Deterministic provider for offline benchmark and regression runs."""

    response_text: str

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            payload = json.loads(self.response_text)
        except json.JSONDecodeError:
            return self.response_text

        if not isinstance(payload, dict):
            return self.response_text

        interaction_id = None
        turn_number = None
        for line in user_prompt.splitlines():
            if line.startswith("interaction_id: "):
                interaction_id = line.removeprefix("interaction_id: ").strip()
            elif line.startswith("turn_number: "):
                turn_value = line.removeprefix("turn_number: ").strip()
                if turn_value.isdigit():
                    turn_number = turn_value

        responses_by_key = payload.get("responses_by_key")
        if isinstance(responses_by_key, dict) and interaction_id is not None and turn_number is not None:
            key = f"{interaction_id}:{turn_number}"
            selected = responses_by_key.get(key)
            if isinstance(selected, dict):
                return json.dumps(selected)

        responses_by_turn = payload.get("responses_by_turn")
        if isinstance(responses_by_turn, dict) and turn_number is not None:
            selected = responses_by_turn.get(turn_number)
            if isinstance(selected, dict):
                return json.dumps(selected)

        default_response = payload.get("default_response")
        if isinstance(default_response, dict):
            return json.dumps(default_response)

        return self.response_text


class LLMObserverTrack(TrackExtractor):
    """LLM observer that normalizes provider output into a canonical JSV hint."""

    track_id = "llm_observer"

    def __init__(self, provider: LLMProvider, model_id: str, prompt_version: str = "llm-observer-v1") -> None:
        self.provider = provider
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.system_prompt = load_prompt("llm_observer_system.txt")
        self.max_normalization_attempts = 4

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
        context_block = ""
        recent = context_turns[-5:]
        if recent:
            lines: list[str] = []
            for ct in recent:
                t = ct.get("turn_number", "?")
                h = ct.get("human_input", "")
                a = ct.get("ai_response", "")
                lines.append(f"[turn {t}] Human: {h[:300]}")
                lines.append(f"[turn {t}] AI: {a[:300]}")
            context_block = "conversation_history:\n" + "\n".join(lines) + "\n\n"
        return (
            f"interaction_id: {interaction_id}\n"
            f"context_module: {context_module}\n"
            f"turn_number: {turn_number}\n\n"
            f"{context_block}"
            f"current_turn_human: {human_input}\n"
            f"current_turn_ai: {ai_response}\n"
        )

    def extract(
        self,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> TrackOutput:
        user_prompt = self._build_user_prompt(
            interaction_id=interaction_id,
            turn_number=turn_number,
            human_input=human_input,
            ai_response=ai_response,
            context_turns=context_turns,
            context_module=context_module,
        )
        raw_response = ""
        last_error: LLMNormalizationError | None = None
        current_prompt = user_prompt
        attempt_responses: list[str] = []
        for attempt in range(self.max_normalization_attempts):
            raw_response = self.provider.generate(system_prompt=self.system_prompt, user_prompt=current_prompt)
            attempt_responses.append(raw_response)
            try:
                normalized = normalize_llm_response(raw_response)
                break
            except LLMNormalizationError as exc:
                last_error = LLMNormalizationError(
                    str(exc),
                    raw_response=raw_response,
                    attempt_responses=list(attempt_responses),
                )
                if attempt + 1 >= self.max_normalization_attempts:
                    raise last_error
                current_prompt = (
                    f"{user_prompt}\n"
                    "previous_response_was_invalid_json: true\n"
                    "repair_instruction: Return one JSON object only. Do not use markdown fences. "
                    "Do not add commentary before or after the JSON. "
                    "The response must start with { and end with }. "
                    "Always include all required fields and at least one evidence span.\n"
                    "required_schema_template:\n"
                    '{'
                    '"judgment_holder":"Human|Shared|AI|Undefined",'
                    '"delegation_awareness":"Explicit|Implicit|Absent",'
                    '"cognitive_engagement":"Active|Reactive|Passive",'
                    '"information_seeking":"Active|Passive|None",'
                    '"confidence":{'
                    '"judgment_holder":"high|medium|low",'
                    '"delegation_awareness":"high|medium|low",'
                    '"cognitive_engagement":"high|medium|low",'
                    '"information_seeking":"high|medium|low"'
                    '},'
                    '"evidence_spans":[{"text":"...","category":"..."}],'
                    '"observer_notes":"..."'
                    '}\n'
                    f"previous_response:\n{raw_response}\n"
                )
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("unreachable normalization state")
        return TrackOutput(
            track_id=self.track_id,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            jsv_hint=normalized["jsv_hint"],
            evidence_spans=normalized["evidence_spans"],
            observer_confidence=normalized.get("observer_confidence"),
            observer_notes=normalized.get("observer_notes", ""),
            raw={"raw_response": raw_response},
        )


def create_env_backed_llm_track() -> LLMObserverTrack:
    provider_kind = os.getenv("JDVP_LLM_PROVIDER", "openai_compatible")
    model = os.getenv("JDVP_LLM_MODEL")

    provider, resolved_model_id = create_env_backed_provider()
    return LLMObserverTrack(provider=provider, model_id=model or resolved_model_id)


def create_env_backed_provider() -> tuple[LLMProvider, str]:
    provider_kind = os.getenv("JDVP_LLM_PROVIDER", "openai_compatible")
    model = os.getenv("JDVP_LLM_MODEL")

    if provider_kind == "openai_compatible":
        base_url = os.getenv("JDVP_LLM_BASE_URL")
        api_key = os.getenv("JDVP_LLM_API_KEY")
        if not base_url or not api_key or not model:
            raise RuntimeError("JDVP_LLM_BASE_URL, JDVP_LLM_API_KEY, and JDVP_LLM_MODEL are required")
        config = LlmConfig(
            provider="openai",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=float(os.getenv("JDVP_LLM_TIMEOUT_SECONDS", "30")),
            prefer_json_mode=os.getenv("JDVP_LLM_PREFER_JSON_MODE", "1") != "0",
            max_tokens=int(os.getenv("JDVP_LLM_MAX_TOKENS")) if os.getenv("JDVP_LLM_MAX_TOKENS") else None,
        )
        client = create_client(config)
        return BufferlineLlmAdapter(client=client, model_id=model), model

    if provider_kind == "static_response":
        response_text = os.getenv("JDVP_LLM_STATIC_RESPONSE")
        response_path = os.getenv("JDVP_LLM_STATIC_RESPONSE_PATH")
        if not response_text and response_path:
            response_text = Path(response_path).read_text(encoding="utf-8")
        if not response_text:
            raise RuntimeError(
                "JDVP_LLM_STATIC_RESPONSE or JDVP_LLM_STATIC_RESPONSE_PATH is required "
                "when JDVP_LLM_PROVIDER=static_response"
            )
        return StaticResponseProvider(response_text=response_text), (model or "static-response")

    raise RuntimeError(f"unsupported JDVP_LLM_PROVIDER: {provider_kind}")
