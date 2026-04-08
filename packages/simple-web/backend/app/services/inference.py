from __future__ import annotations

import pickle
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]
DA_WEIGHTS = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570, "information_dependency": 0.268}


class InferenceService:
    def __init__(self, model_path: str, regressors_path: str, backend: str = "onnx"):
        kwargs = {}
        if backend == "onnx":
            kwargs["backend"] = "onnx"
            kwargs["model_kwargs"] = {"file_name": "model.onnx"}
        self.embedder = SentenceTransformer(model_path, **kwargs)
        with open(regressors_path, "rb") as f:
            data = pickle.load(f)
        self.regressors = data["regressors"]
        self.n_training_turns = data["n_training_turns"]

    def predict(self, user_turns: list[str]) -> list[dict]:
        texts = self._build_texts(user_turns)
        X = self.embedder.encode(texts, batch_size=64, show_progress_bar=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = normalize(X)

        predictions = {}
        for axis in AXES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                predictions[axis] = np.clip(self.regressors[axis].predict(X), 0, 10)

        predictions["da_derived"] = sum(
            DA_WEIGHTS[a] * predictions[a] for a in AXES
        )

        results = []
        for i in range(len(user_turns)):
            results.append({
                "turn_number": i,
                "human_input_preview": user_turns[i][:100],
                "scores": {
                    field: round(float(predictions[field][i]), 2)
                    for field in AXES + ["da_derived"]
                },
            })
        return results

    def _build_texts(self, user_turns: list[str]) -> list[str]:
        texts = []
        for i, turn in enumerate(user_turns):
            history = user_turns[max(0, i - 3):i]
            history_text = ""
            if history:
                history_text = " | ".join(h[-200:] for h in history) + " || "
            texts.append(history_text + f"Human: {turn}")
        return texts
