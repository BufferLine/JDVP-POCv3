"""ONNX-based inference service — drop-in replacement for inference.py.

Removes PyTorch/sentence-transformers dependency.
Requires only: onnxruntime, tokenizers, numpy, scikit-learn.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]
DA_WEIGHTS = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570, "information_dependency": 0.268}


class InferenceServiceONNX:
    def __init__(self, onnx_model_dir: str, regressors_path: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(onnx_model_dir)

        # Load ONNX model (check multiple locations)
        candidates = [
            model_dir / "model.onnx",
            model_dir / "onnx" / "model.onnx",
            model_dir / "model_optimized.onnx",
        ]
        onnx_path = None
        for c in candidates:
            if c.exists():
                onnx_path = c
                break
        if onnx_path is None:
            found = list(model_dir.rglob("*.onnx"))
            if found:
                onnx_path = found[0]
            else:
                raise FileNotFoundError(f"No .onnx file found in {model_dir}")

        self.session = ort.InferenceSession(str(onnx_path))

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        self.tokenizer.enable_truncation(max_length=256)

        # Load pooling config
        pooling_config = model_dir / "1_Pooling" / "config.json"
        if pooling_config.exists():
            self.pooling = json.loads(pooling_config.read_text())
        else:
            self.pooling = {"pooling_mode_mean_tokens": True}

        # Load regressors
        with open(regressors_path, "rb") as f:
            data = pickle.load(f)
        self.regressors = data["regressors"]
        self.n_training_turns = data["n_training_turns"]

    def _encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Filter to only inputs the model expects
        model_inputs = {i.name for i in self.session.get_inputs()}
        inputs = {k: v for k, v in inputs.items() if k in model_inputs}

        outputs = self.session.run(None, inputs)
        hidden_states = outputs[0]  # (batch, seq, dim)

        # Pooling (read from config)
        if self.pooling.get("pooling_mode_cls_token"):
            embeddings = hidden_states[:, 0, :]
        else:
            # Mean pooling fallback
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = (hidden_states * mask_expanded).sum(axis=1)
            sum_mask = mask_expanded.sum(axis=1).clip(min=1e-9)
            embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        return embeddings / norms

    def predict(self, user_turns: list[str]) -> list[dict]:
        texts = self._build_texts(user_turns)
        X = self._encode(texts)
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
