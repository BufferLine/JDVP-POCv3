from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.dataset.build_fewshot_pack import build_fewshot_pack
from src.method.tracks.cheap_ml_baseline import CheapMLBaselineTrack, create_env_backed_cheap_ml_track
from src.pipeline.run_poc import run_poc


ROOT = Path(__file__).resolve().parents[1]


class CheapMLBaselineTests(unittest.TestCase):
    def test_cheap_ml_track_predicts_valid_jsv_fields(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root,
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=3)

            track = CheapMLBaselineTrack(fewshot_pack_path=pack_path)
            output = track.extract(
                interaction_id="session-1",
                turn_number=0,
                human_input="I need help deciding whether this option fits my priorities.",
                ai_response="Tell me the tradeoffs you care about and I can help compare them.",
                context_turns=[],
                context_module="general",
            )
            self.assertEqual(output.track_id, "cheap_ml_baseline")
            self.assertIn(output.jsv_hint["judgment_holder"], {"Human", "Shared", "AI"})
            self.assertIn("confidence", output.jsv_hint)

    def test_env_backed_cheap_ml_track_uses_fewshot_pack(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root,
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=3)
            previous = os.environ.get("JDVP_FEWSHOT_PACK_PATH")
            try:
                os.environ["JDVP_FEWSHOT_PACK_PATH"] = str(pack_path)
                track = create_env_backed_cheap_ml_track()
            finally:
                if previous is None:
                    os.environ.pop("JDVP_FEWSHOT_PACK_PATH", None)
                else:
                    os.environ["JDVP_FEWSHOT_PACK_PATH"] = previous

            self.assertEqual(track.track_id, "cheap_ml_baseline")

    def test_cheap_ml_track_can_write_and_reload_model_artifact(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root,
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=3)
            artifact_path = output_root / "cheap-ml-model.json"

            trained_track = CheapMLBaselineTrack(fewshot_pack_path=pack_path)
            trained_track.write_artifact(artifact_path)
            loaded_track = CheapMLBaselineTrack.from_model_artifact(artifact_path)

            output = loaded_track.extract(
                interaction_id="session-1",
                turn_number=0,
                human_input="I need help deciding whether this option fits my priorities.",
                ai_response="Tell me the tradeoffs you care about and I can help compare them.",
                context_turns=[],
                context_module="general",
            )
            self.assertEqual(output.track_id, "cheap_ml_baseline")
            self.assertEqual(output.raw["model_artifact_path"], str(artifact_path))

    def test_env_backed_cheap_ml_track_prefers_model_artifact(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            run_dir = run_poc(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root,
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=run_dir, output_path=pack_path, max_examples=3)
            artifact_path = output_root / "cheap-ml-model.json"
            CheapMLBaselineTrack(fewshot_pack_path=pack_path).write_artifact(artifact_path)

            previous_pack = os.environ.get("JDVP_FEWSHOT_PACK_PATH")
            previous_model = os.environ.get("JDVP_CHEAP_ML_MODEL_PATH")
            try:
                os.environ["JDVP_FEWSHOT_PACK_PATH"] = str(pack_path)
                os.environ["JDVP_CHEAP_ML_MODEL_PATH"] = str(artifact_path)
                track = create_env_backed_cheap_ml_track()
            finally:
                if previous_pack is None:
                    os.environ.pop("JDVP_FEWSHOT_PACK_PATH", None)
                else:
                    os.environ["JDVP_FEWSHOT_PACK_PATH"] = previous_pack
                if previous_model is None:
                    os.environ.pop("JDVP_CHEAP_ML_MODEL_PATH", None)
                else:
                    os.environ["JDVP_CHEAP_ML_MODEL_PATH"] = previous_model

            self.assertEqual(track.raw_model_id if hasattr(track, "raw_model_id") else track.model_id, "naive-bayes-v1")


if __name__ == "__main__":
    unittest.main()
