from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset.generate_dataset import generate_dataset
from src.pipeline.run_poc import run_poc


ROOT = Path(__file__).resolve().parents[1]


class DatasetGenerationTests(unittest.TestCase):
    def test_dataset_generation_writes_manifest_and_splits(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=scenario_pack,
                count_per_scenario=2,
                seed=11,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            splits = json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["dataset_id"], "generated/synthetic-general/v1")
            self.assertEqual(manifest["scenario_pack_id"], "general-scenarios-v1")
            self.assertEqual(manifest["item_count"], 6)
            self.assertEqual(len(manifest["items"]), 6)
            self.assertEqual(sum(len(values) for values in splits.values()), 6)
            self.assertTrue((dataset_root / "interactions").is_dir())
            self.assertIn("turn_variant_choices", manifest["items"][0])
            self.assertEqual(len(manifest["items"][0]["turn_variant_choices"]), 3)

    def test_generated_dataset_can_feed_pipeline(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=output_root,
                scenario_pack_path=scenario_pack,
                count_per_scenario=1,
                seed=5,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction_path = dataset_root / first_item["relative_path"]
            run_dir = run_poc(
                input_path=interaction_path,
                run_id="generated-smoke",
                output_root=output_root / "runs",
                track_name="fixture_hint",
            )
            self.assertTrue((run_dir / "canonical" / "trajectory.json").is_file())
            trajectory = json.loads((run_dir / "canonical" / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory["interaction_id"], first_item["interaction_id"])

    def test_dataset_generation_uses_turn_variants(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=scenario_pack,
                count_per_scenario=3,
                seed=11,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            travel_items = [
                item for item in manifest["items"]
                if item["scenario_id"] == "travel-planning"
            ]
            rendered_inputs = {
                json.loads((dataset_root / item["relative_path"]).read_text(encoding="utf-8"))["turns"][0]["human_input"]
                for item in travel_items
            }
            self.assertGreater(len(rendered_inputs), 1)

    def test_dataset_generation_supports_blueprint_selection(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v2.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = generate_dataset(
                dataset_name="synthetic-general-rich",
                dataset_version="v1",
                output_root=Path(tmp_dir),
                scenario_pack_path=scenario_pack,
                count_per_scenario=4,
                seed=11,
            )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scenario_pack_id"], "general-scenarios-v2")
            self.assertTrue(all("blueprint_id" in item for item in manifest["items"]))
            blueprint_ids = {item["blueprint_id"] for item in manifest["items"]}
            self.assertGreater(len(blueprint_ids), 1)
            turn_counts = {item["turn_count"] for item in manifest["items"]}
            self.assertIn(4, turn_counts)

    def test_dataset_generation_supports_llm_materialization(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        llm_response = json.dumps(
            {
                "turns": [
                    {
                        "turn_number": 0,
                        "human_input": "I want help deciding whether this trip is worth it.",
                        "ai_response": "Tell me what matters most and I will help compare the options.",
                    },
                    {
                        "turn_number": 1,
                        "human_input": "Compare the cost and convenience for me.",
                        "ai_response": "I can lay out the tradeoffs so you can see which option fits better.",
                    },
                    {
                        "turn_number": 2,
                        "human_input": "Just give me your recommendation.",
                        "ai_response": "I would choose the option that best fits your stated priorities.",
                    },
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(type("StubProvider", (), {"generate": lambda self, **_: llm_response})(), "stub-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-llm",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm",
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["generation"]["mode"], "llm")
            self.assertEqual(manifest["generation"]["llm_model_id"], "stub-model")
            self.assertEqual(interaction["turns"][0]["human_input"], "I want help deciding whether this trip is worth it.")
            self.assertIn("jsv_hint", interaction["turns"][0]["meta"])

    def test_dataset_generation_supports_llm_turn_simulation(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"

        class StubTurnSimProvider:
            def __init__(self) -> None:
                self.human_calls = 0
                self.ai_calls = 0

            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                if "human side" in system_prompt:
                    value = self.human_calls
                    self.human_calls += 1
                    payload = {"human_input": f"human turn {value} based on prior context"}
                    return json.dumps(payload)
                value = self.ai_calls
                self.ai_calls += 1
                payload = {"ai_response": f"assistant turn {value} responding to the user"}
                return json.dumps(payload)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(StubTurnSimProvider(), "stub-turn-sim-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-turn-sim",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm_turn_simulated",
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["generation"]["mode"], "llm_turn_simulated")
            self.assertEqual(manifest["generation"]["llm_model_id"], "stub-turn-sim-model")
            self.assertEqual(interaction["turns"][0]["human_input"], "human turn 0 based on prior context")
            self.assertEqual(interaction["turns"][1]["ai_response"], "assistant turn 1 responding to the user")

    def test_llm_turn_simulation_retries_invalid_json(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"

        class FlakyTurnSimProvider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "not json at all"
                if "human side" in system_prompt:
                    return json.dumps({"human_input": "recovered human turn"})
                return json.dumps({"ai_response": "recovered ai turn"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(FlakyTurnSimProvider(), "stub-flaky-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-turn-retry",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm_turn_simulated",
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(interaction["turns"][0]["human_input"], "recovered human turn")

    def test_llm_turn_simulation_salvages_truncated_json(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"

        class TruncatedTurnSimProvider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return '{\n  "human_input": "truncated but usable"'
                if "human side" in system_prompt:
                    return json.dumps({"human_input": "normal human turn"})
                return json.dumps({"ai_response": "normal ai turn"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(TruncatedTurnSimProvider(), "stub-truncated-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-turn-truncated",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm_turn_simulated",
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(interaction["turns"][0]["human_input"], "truncated but usable")

    def test_dataset_generation_reruns_only_failed_items(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"

        class FailOnceProvider:
            def __init__(self) -> None:
                self.failed = False
                self.call_count = 0

            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                self.call_count += 1
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("transient generation failure")
                if "human side" in system_prompt:
                    return json.dumps({"human_input": f"human call {self.call_count}"})
                return json.dumps({"ai_response": f"ai call {self.call_count}"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            provider = FailOnceProvider()
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                with patch(
                    "src.dataset.generate_dataset.create_env_backed_provider",
                    return_value=(provider, "resume-model"),
                ):
                    with self.assertRaises(RuntimeError):
                        generate_dataset(
                            dataset_name="synthetic-general-resume",
                            dataset_version="v2",
                            output_root=Path(tmp_dir),
                            scenario_pack_path=scenario_pack,
                            count_per_scenario=1,
                            seed=11,
                            generation_mode="llm_turn_simulated",
                        )
                    first_call_count = provider.call_count
                    dataset_root = generate_dataset(
                        dataset_name="synthetic-general-resume",
                        dataset_version="v2",
                        output_root=Path(tmp_dir),
                        scenario_pack_path=scenario_pack,
                        count_per_scenario=1,
                        seed=11,
                        generation_mode="llm_turn_simulated",
                    )
                manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
                progress = json.loads((dataset_root / "generation_progress.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["item_count"], 3)
                self.assertTrue(progress["is_complete"])
                self.assertLess(provider.call_count - first_call_count, 18)
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
