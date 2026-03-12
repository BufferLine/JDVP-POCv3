"""Provider-backed LLM observer track."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
    if provider_kind != "openai_compatible":
        raise RuntimeError(f"unsupported JDVP_LLM_PROVIDER: {provider_kind}")
    base_url = os.getenv("JDVP_LLM_BASE_URL")
    api_key = os.getenv("JDVP_LLM_API_KEY")
    model = os.getenv("JDVP_LLM_MODEL")
    if not base_url or not api_key or not model:
        raise RuntimeError("JDVP_LLM_BASE_URL, JDVP_LLM_API_KEY, and JDVP_LLM_MODEL are required")
    provider = OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key)
    return LLMObserverTrack(provider=provider, model_id=model)
