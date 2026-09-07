"""Calibrated four-axis JDVP observer over a local embedding encoder."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.protocol_core.enums import CORE_FIELD_NAMES

from .base import TrackExtractor, TrackOutput
from .embedding_screen import _turn_text, _unit_rows


ARTIFACT_SCHEMA_VERSION = "jdvp-embedding-calibrator-v1"


class EmbeddingCalibratedTrack(TrackExtractor):
    """Apply persisted per-axis linear calibration heads to local embeddings."""

    track_id = "embedding_calibrated"

    def __init__(self, *, artifact_path: Path, embedder: Any | None = None, embedding_model_path: Path | None = None) -> None:
        self.artifact_path = artifact_path
        self.artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if self.artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unexpected embedding calibrator artifact: {self.artifact.get('schema_version')}")
        model_path = embedding_model_path or Path(self.artifact["embedding_model_path"])
        self.embedding_model_path = model_path
        self.model_id = str(self.artifact["model_id"])
        self.prompt_version = "embedding-calibrated-v1"
        self.embedder = embedder or self._load_embedder(model_path)
        self.heads = self.artifact["heads"]
        self.metrics = self.artifact.get("holdout_metrics", {})

    @staticmethod
    def _load_embedder(embedding_model_path: Path) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError("embedding_calibrated requires sentence-transformers") from exc
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
        vector = _unit_rows(self.embedder.encode([_turn_text(human_input, ai_response)], show_progress_bar=False))[0]
        continuous_scores: dict[str, float] = {}
        jsv_hint: dict[str, Any] = {}
        maes: list[float] = []
        for field_name in CORE_FIELD_NAMES:
            head = self.heads[field_name]
            score = float(np.dot(vector, np.asarray(head["coef"], dtype=float)) + float(head["intercept"]))
            score = max(0.0, min(10.0, score))
            continuous_scores[field_name] = round(score, 4)
            jsv_hint[field_name] = int(round(score))
            maes.append(float(self.metrics.get(field_name, {}).get("mae", 5.0)))

        mean_mae = sum(maes) / len(maes)
        confidence_value = max(0.0, min(1.0, 1.0 - (mean_mae / 10.0)))
        confidence = "high" if confidence_value >= 0.85 else "medium" if confidence_value >= 0.65 else "low"
        jsv_hint["confidence"] = {field_name: confidence for field_name in CORE_FIELD_NAMES}

        return TrackOutput(
            track_id=self.track_id,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            jsv_hint=jsv_hint,
            evidence_spans=[{"text": human_input[:160], "category": "embedding_calibrated_query"}],
            observer_confidence=round(confidence_value, 4),
            observer_notes=(
                "Embedding scores calibrated on the artifact's declared label source; "
                "not a validated critical-thinking assessment."
            ),
            raw={
                "calibrator_artifact_path": str(self.artifact_path),
                "embedding_model_path": str(self.embedding_model_path),
                "continuous_scores": continuous_scores,
                "holdout_metrics": self.metrics,
                "context_turn_count": len(context_turns),
                "context_module": context_module,
            },
        )


def create_env_backed_embedding_calibrated_track() -> EmbeddingCalibratedTrack:
    artifact_path = os.getenv("JDVP_EMBEDDING_CALIBRATOR_PATH")
    if not artifact_path:
        raise RuntimeError("JDVP_EMBEDDING_CALIBRATOR_PATH is required")
    model_path = os.getenv("JDVP_EMBEDDING_MODEL_PATH")
    return EmbeddingCalibratedTrack(
        artifact_path=Path(artifact_path),
        embedding_model_path=Path(model_path) if model_path else None,
    )
