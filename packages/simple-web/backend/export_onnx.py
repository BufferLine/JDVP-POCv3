#!/usr/bin/env python3
"""Export embedding model to ONNX format and re-export regressors.

Usage:
    python3 export_onnx.py --model-path ../../models/jdvp-embedding-bge-small-en-v1.5-20260408 \
        --output-dir ./onnx_model
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="onnx_model")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path))

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting ONNX to {onnx_path}...")

    # sentence-transformers has built-in ONNX export via optimum
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        # Export via optimum
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path), export=True
        )
        ort_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("Exported via optimum")
    except ImportError:
        # Fallback: manual ONNX export
        import torch
        import onnx

        auto_model = model[0].auto_model
        tokenizer_obj = model.tokenizer

        dummy = tokenizer_obj("hello world", return_tensors="pt", padding=True, truncation=True)
        dummy_input = {k: v for k, v in dummy.items()}

        torch.onnx.export(
            auto_model,
            tuple(dummy_input.values()),
            str(onnx_path),
            input_names=list(dummy_input.keys()),
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=14,
        )
        # Copy tokenizer files
        for f in model_path.glob("tokenizer*"):
            shutil.copy2(f, output_dir / f.name)
        if (model_path / "special_tokens_map.json").exists():
            shutil.copy2(model_path / "special_tokens_map.json", output_dir)
        print("Exported via torch.onnx.export")

    # Copy pooling config
    pooling_dir = model_path / "1_Pooling"
    if pooling_dir.exists():
        out_pooling = output_dir / "1_Pooling"
        out_pooling.mkdir(exist_ok=True)
        for f in pooling_dir.iterdir():
            shutil.copy2(f, out_pooling / f.name)

    # Save metadata
    meta = {
        "source_model": str(model_path),
        "format": "onnx",
        "embedding_dim": model.get_sentence_embedding_dimension(),
    }
    (output_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))

    # Check sizes
    orig_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1024 / 1024
    onnx_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\nOriginal: {orig_size:.0f} MB")
    print(f"ONNX:     {onnx_size:.0f} MB")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
