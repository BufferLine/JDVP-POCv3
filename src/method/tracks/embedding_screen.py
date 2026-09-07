"""Embedding-based nearest-prototype screening track.

This is intentionally a screening observer, not a validated assessment model.
It retrieves the closest labeled prototype in embedding space and reuses that
prototype's canonical JDVP levels through the normal extraction contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.method.fewshot.selector import load_fewshot_pack
from src.protocol_core.enums import CORE_FIELD_NAMES, normalize_core_level

from .base import TrackExtractor, TrackOutput


def _turn_text(human_input: str, ai_response: str) -> str:
    return f"Human: {human_input}\nAI: {ai_response}"


def _unit_rows(values: Any) -> np.ndarray:
    rows = np.asarray(values, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


class EmbeddingScreenTrack(TrackExtractor):
    """Local embedding retrieval over explicitly supplied labeled prototypes."""

    track_id = "embedding_screen"

    def __init__(
        self,
        *,
        prototype_pack_path: Path,
        embedding_model_path: Path,
        embedder: Any | None = None,
    ) -> None:
        self.prototype_pack_path = prototype_pack_path
        self.embedding_model_path = embedding_model_path
        self.model_id = f"embedding-prototype:{embedding_model_path.name}"
        self.prompt_version = "embedding-screen-v1"
        pack = load_fewshot_pack(prototype_pack_path)
        self.prototypes = list(pack.get("examples", []))
        if not self.prototypes:
            raise ValueError("embedding_screen requires a non-empty prototype pack")
        self.embedder = embedder or self._load_embedder(embedding_model_path)
        self.prototype_vectors = _unit_rows(
            self.embedder.encode(
                [_turn_text(example["human_input"], example["ai_response"]) for example in self.prototypes],
                show_progress_bar=False,
            )
        )

    @staticmethod
    def _load_embedder(embedding_model_path: Path) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError("embedding_screen requires sentence-transformers") from exc
        try:
            return SentenceTransformer(str(embedding_model_path), local_files_only=True)
        except OSError as exc:
            raise RuntimeError(
                f"unable to load local embedding model at {embedding_model_path}; "
                "supply a complete model directory including its weights"
            ) from exc

    def extract(
        self,
        interaction_id: str,
        turn_number: int,
        human_input: str,
        ai_response: str,
        context_turns: list[dict[str, Any]],
        context_module: str,
    ) -> TrackOutput:
        query_vector = _unit_rows(
            self.embedder.encode([_turn_text(human_input, ai_response)], show_progress_bar=False)
        )[0]
        similarities = self.prototype_vectors @ query_vector
        prototype_index = int(np.argmax(similarities))
        similarity = float(similarities[prototype_index])
        prototype = self.prototypes[prototype_index]

        jsv_hint = {
            field_name: normalize_core_level(field_name, prototype["jsv_hint"][field_name])
            for field_name in CORE_FIELD_NAMES
        }
        confidence = "high" if similarity >= 0.8 else "medium" if similarity >= 0.55 else "low"
        jsv_hint["confidence"] = {field_name: confidence for field_name in CORE_FIELD_NAMES}

        return TrackOutput(
            track_id=self.track_id,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            jsv_hint=jsv_hint,
            evidence_spans=[{"text": human_input[:160], "category": "embedding_query"}],
            observer_confidence=round(max(0.0, min(1.0, (similarity + 1.0) / 2.0)), 4),
            observer_notes="Embedding nearest-prototype screening; not a validated assessment model.",
            raw={
                "embedding_model_path": str(self.embedding_model_path),
                "prototype_pack_path": str(self.prototype_pack_path),
                "nearest_prototype_interaction_id": prototype["interaction_id"],
                "nearest_prototype_turn_number": prototype["turn_number"],
                "cosine_similarity": round(similarity, 6),
                "context_turn_count": len(context_turns),
                "context_module": context_module,
            },
        )


def create_env_backed_embedding_screen_track() -> EmbeddingScreenTrack:
    model_path = os.getenv("JDVP_EMBEDDING_MODEL_PATH")
    prototype_pack_path = os.getenv("JDVP_EMBEDDING_PROTOTYPE_PACK_PATH")
    if not model_path or not prototype_pack_path:
        raise RuntimeError(
            "JDVP_EMBEDDING_MODEL_PATH and JDVP_EMBEDDING_PROTOTYPE_PACK_PATH are required"
        )
    return EmbeddingScreenTrack(
        embedding_model_path=Path(model_path),
        prototype_pack_path=Path(prototype_pack_path),
    )
