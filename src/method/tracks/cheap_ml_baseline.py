"""Very lightweight Naive Bayes baseline trained from a few-shot pack."""

from __future__ import annotations

import math
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.method.fewshot.selector import load_fewshot_pack
from src.protocol_core.jsv_types import CORE_FIELD_NAMES
from src.shared_utils import tokenize as _tokenize

from .base import TrackExtractor, TrackOutput


@dataclass(frozen=True)
class _FieldModel:
    class_doc_counts: dict[str, int]
    class_token_counts: dict[str, Counter[str]]
    class_total_tokens: dict[str, int]
    vocabulary: set[str]

    def predict(self, tokens: list[str]) -> tuple[str, float]:
        total_docs = sum(self.class_doc_counts.values())
        best_label = ""
        best_score = float("-inf")
        second_best = float("-inf")
        vocab_size = max(len(self.vocabulary), 1)

        for label, doc_count in self.class_doc_counts.items():
            score = math.log(doc_count / total_docs)
            token_counts = self.class_token_counts[label]
            total_tokens = self.class_total_tokens[label]
            for token in tokens:
                score += math.log((token_counts[token] + 1) / (total_tokens + vocab_size))
            if score > best_score:
                second_best = best_score
                best_score = score
                best_label = label
            elif score > second_best:
                second_best = score

        margin = best_score - second_best if second_best != float("-inf") else 1.0
        return best_label, margin

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_doc_counts": self.class_doc_counts,
            "class_token_counts": {
                label: dict(counter)
                for label, counter in self.class_token_counts.items()
            },
            "class_total_tokens": self.class_total_tokens,
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_FieldModel":
        return cls(
            class_doc_counts={str(key): int(value) for key, value in payload["class_doc_counts"].items()},
            class_token_counts={
                str(label): Counter({str(token): int(count) for token, count in counts.items()})
                for label, counts in payload["class_token_counts"].items()
            },
            class_total_tokens={str(key): int(value) for key, value in payload["class_total_tokens"].items()},
            vocabulary={str(token) for token in payload["vocabulary"]},
        )


def _margin_to_confidence(margin: float) -> str:
    if margin >= 1.5:
        return "high"
    if margin >= 0.75:
        return "medium"
    return "low"


class CheapMLBaselineTrack(TrackExtractor):
    """Naive Bayes classifier over few-shot pack examples."""

    track_id = "cheap_ml_baseline"

    def __init__(
        self,
        *,
        fewshot_pack_path: Path | None,
        field_models: dict[str, _FieldModel] | None = None,
        model_artifact_path: Path | None = None,
        model_id: str = "naive-bayes-v1",
    ) -> None:
        self.fewshot_pack_path = fewshot_pack_path
        self.model_artifact_path = model_artifact_path
        self.model_id = model_id
        self.prompt_version = "cheap-ml-v1"
        if field_models is not None:
            self.field_models = field_models
        else:
            if fewshot_pack_path is None:
                raise RuntimeError("fewshot_pack_path is required when field_models are not provided")
            pack = load_fewshot_pack(fewshot_pack_path)
            self.field_models = self._train_models(pack["examples"])

    def _train_models(self, examples: list[dict[str, Any]]) -> dict[str, _FieldModel]:
        if not examples:
            raise ValueError("cannot train cheap_ml_baseline from an empty fewshot pack")
        field_doc_counts: dict[str, dict[str, int]] = {field: defaultdict(int) for field in CORE_FIELD_NAMES}
        field_token_counts: dict[str, dict[str, Counter[str]]] = {
            field: defaultdict(Counter) for field in CORE_FIELD_NAMES
        }
        field_total_tokens: dict[str, dict[str, int]] = {field: defaultdict(int) for field in CORE_FIELD_NAMES}
        field_vocabularies: dict[str, set[str]] = {field: set() for field in CORE_FIELD_NAMES}

        for example in examples:
            tokens = _tokenize(f"{example['human_input']} {example['ai_response']}")
            for field in CORE_FIELD_NAMES:
                label = str(example["jsv_hint"][field])
                field_doc_counts[field][label] += 1
                field_token_counts[field][label].update(tokens)
                field_total_tokens[field][label] += len(tokens)
                field_vocabularies[field].update(tokens)

        return {
            field: _FieldModel(
                class_doc_counts=dict(field_doc_counts[field]),
                class_token_counts=dict(field_token_counts[field]),
                class_total_tokens=dict(field_total_tokens[field]),
                vocabulary=field_vocabularies[field],
            )
            for field in CORE_FIELD_NAMES
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "pocv3-cheap-ml-model-v1",
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "fewshot_pack_path": str(self.fewshot_pack_path) if self.fewshot_pack_path is not None else None,
            "field_models": {
                field_name: model.to_dict()
                for field_name, model in self.field_models.items()
            },
        }

    def write_artifact(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_artifact_dict(), indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def from_model_artifact(cls, model_artifact_path: Path) -> "CheapMLBaselineTrack":
        payload = json.loads(model_artifact_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "pocv3-cheap-ml-model-v1":
            raise ValueError(f"unexpected cheap ML artifact schema: {payload.get('schema_version')}")
        field_models = {
            str(field_name): _FieldModel.from_dict(field_payload)
            for field_name, field_payload in payload["field_models"].items()
        }
        missing_fields = set(CORE_FIELD_NAMES) - set(field_models.keys())
        if missing_fields:
            raise ValueError(f"cheap ML artifact missing field models: {sorted(missing_fields)}")
        fewshot_pack_path = payload.get("fewshot_pack_path")
        return cls(
            fewshot_pack_path=Path(fewshot_pack_path) if fewshot_pack_path else None,
            field_models=field_models,
            model_artifact_path=model_artifact_path,
            model_id=str(payload.get("model_id", "naive-bayes-v1")),
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
        tokens = _tokenize(f"{human_input} {ai_response}")
        jsv_hint: dict[str, Any] = {}
        confidence: dict[str, str] = {}
        for field_name in CORE_FIELD_NAMES:
            label, margin = self.field_models[field_name].predict(tokens)
            jsv_hint[field_name] = label
            confidence[field_name] = _margin_to_confidence(margin)
        jsv_hint["confidence"] = confidence

        evidence_tokens = tokens[: min(4, len(tokens))]
        evidence_spans = (
            [{"text": " ".join(evidence_tokens), "category": "cheap_ml_tokens"}]
            if evidence_tokens
            else [{"text": human_input[:160], "category": "cheap_ml_excerpt"}]
        )

        return TrackOutput(
            track_id=self.track_id,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            jsv_hint=jsv_hint,
            evidence_spans=evidence_spans,
            observer_confidence=0.55,
            observer_notes="Naive Bayes prediction from few-shot pack",
            raw={
                "fewshot_pack_path": str(self.fewshot_pack_path) if self.fewshot_pack_path is not None else None,
                "model_artifact_path": (
                    str(self.model_artifact_path)
                    if self.model_artifact_path is not None
                    else None
                ),
                "token_count": len(tokens),
                "context_turn_count": len(context_turns),
                "context_module": context_module,
            },
        )


def create_env_backed_cheap_ml_track() -> CheapMLBaselineTrack:
    model_artifact_path = os.getenv("JDVP_CHEAP_ML_MODEL_PATH")
    if model_artifact_path:
        return CheapMLBaselineTrack.from_model_artifact(Path(model_artifact_path))
    fewshot_pack_path = os.getenv("JDVP_FEWSHOT_PACK_PATH")
    if not fewshot_pack_path:
        raise RuntimeError("JDVP_CHEAP_ML_MODEL_PATH or JDVP_FEWSHOT_PACK_PATH is required")
    return CheapMLBaselineTrack(fewshot_pack_path=Path(fewshot_pack_path))
