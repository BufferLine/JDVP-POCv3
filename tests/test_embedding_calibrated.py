from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.method.tracks.embedding_calibrated import ARTIFACT_SCHEMA_VERSION, EmbeddingCalibratedTrack
from src.protocol_core.enums import CORE_FIELD_NAMES


class _FakeEmbedder:
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        del texts, show_progress_bar
        return np.asarray([[1.0, 0.0]])


class EmbeddingCalibratedTests(unittest.TestCase):
    def test_calibrated_track_returns_four_bounded_integer_scores(self) -> None:
        heads = {
            field_name: {"coef": [float(index + 1), 0.0], "intercept": 1.5}
            for index, field_name in enumerate(CORE_FIELD_NAMES)
        }
        artifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": "test-calibrator",
            "embedding_model_path": "/models/demo",
            "heads": heads,
            "holdout_metrics": {field_name: {"mae": 2.0} for field_name in CORE_FIELD_NAMES},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "calibrator.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            track = EmbeddingCalibratedTrack(artifact_path=artifact_path, embedder=_FakeEmbedder())
            output = track.extract("session-1", 0, "Help me decide.", "Let's compare options.", [], "general")

        self.assertEqual(output.track_id, "embedding_calibrated")
        self.assertEqual(output.jsv_hint["judgment_holder"], 2)
        self.assertEqual(output.jsv_hint["information_seeking"], 6)
        self.assertTrue(all(0 <= output.jsv_hint[field_name] <= 10 for field_name in CORE_FIELD_NAMES))
        self.assertIn("continuous_scores", output.raw)


if __name__ == "__main__":
    unittest.main()
