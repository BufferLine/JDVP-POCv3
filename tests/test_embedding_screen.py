from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.method.tracks.embedding_screen import EmbeddingScreenTrack


ROOT = Path(__file__).resolve().parents[1]


class _FakeEmbedder:
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        del show_progress_bar
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                1.0 if "job offer" in lowered else 0.0,
                1.0 if "compare salary" in lowered else 0.0,
                1.0 if "what would you choose" in lowered else 0.0,
            ])
        return np.asarray(vectors)


class EmbeddingScreenTests(unittest.TestCase):
    def test_embedding_screen_returns_nearest_prototype_labels(self) -> None:
        track = EmbeddingScreenTrack(
            prototype_pack_path=ROOT / "data" / "fewshot" / "general-fixture-pack-v1.json",
            embedding_model_path=Path("/models/demo"),
            embedder=_FakeEmbedder(),
        )
        output = track.extract(
            interaction_id="new-session",
            turn_number=0,
            human_input="Given that, what would you choose if you were me?",
            ai_response="I would lean toward the role with better long-term growth.",
            context_turns=[],
            context_module="general",
        )

        self.assertEqual(output.track_id, "embedding_screen")
        self.assertEqual(output.jsv_hint["judgment_holder"], 9)
        self.assertEqual(output.jsv_hint["delegation_awareness"], 5)
        self.assertEqual(output.raw["nearest_prototype_turn_number"], 2)


if __name__ == "__main__":
    unittest.main()
