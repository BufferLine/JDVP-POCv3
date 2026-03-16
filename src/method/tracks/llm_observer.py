"""Provider-backed LLM observer track."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

from src.method.evidence.prompt_loader import load_prompt
from src.method.normalization.llm_response import LLMNormalizationError, normalize_llm_response

from .base import TrackExtractor, TrackOutput


class LLMProvider(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat-completions client."""

    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 30.0

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        req = request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"provider request failed: {exc}") from exc
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("provider response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("provider response missing message content")
        return content


@dataclass
class StaticResponseProvider:
    """Deterministic provider for offline benchmark and regression runs."""

    response_text: str

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response_text


class LLMObserverTrack(TrackExtractor):
    """LLM observer that normalizes provider output into a canonical JSV hint."""

    track_id = "llm_observer"

    def __init__(self, provider: LLMProvider, model_id: str, prompt_version: str = "llm-observer-v1") -> None:
        self.provider = provider
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.system_prompt = load_prompt("llm_observer_system.txt")

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
        context_excerpt = json.dumps(context_turns[-3:], ensure_ascii=False)
        return (
            f"interaction_id: {interaction_id}\n"
            f"context_module: {context_module}\n"
            f"turn_number: {turn_number}\n"
            f"human_input: {human_input}\n"
            f"ai_response: {ai_response}\n"
            f"recent_context: {context_excerpt}\n"
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
        raw_response = self.provider.generate(system_prompt=self.system_prompt, user_prompt=user_prompt)
        normalized = normalize_llm_response(raw_response)
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
        provider: LLMProvider = OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key)
        return provider, model

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
