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


def _write_single_scenario_pack(*, tmp_dir: str, scenario_pack: Path) -> Path:
    payload = json.loads(scenario_pack.read_text(encoding="utf-8"))
    payload["scenarios"] = [payload["scenarios"][0]]
    out_path = Path(tmp_dir) / "single-scenario-pack.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path


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
                    payload = {
                        "human_input": [
                            "I am trying to think this through and want help comparing the tradeoffs.",
                            "Can you walk me through the difference between the two main factors?",
                            "If you had to recommend one direction, which would you take?",
                        ][value]
                    }
                    return json.dumps(payload)
                value = self.ai_calls
                self.ai_calls += 1
                payload = {
                    "ai_response": [
                        "Yes. Tell me which tradeoffs matter most and I can help structure the comparison.",
                        "I can compare those factors and show where each option wins or loses.",
                        "I would recommend the option that best protects your priorities while keeping the upside.",
                    ][value]
                }
                return json.dumps(payload)

        with tempfile.TemporaryDirectory() as tmp_dir:
            single_scenario_pack = _write_single_scenario_pack(tmp_dir=tmp_dir, scenario_pack=scenario_pack)
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(StubTurnSimProvider(), "stub-turn-sim-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-turn-sim",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=single_scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm_turn_simulated",
                    enable_quality_gate=False,
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["generation"]["mode"], "llm_turn_simulated")
            self.assertEqual(manifest["generation"]["llm_model_id"], "stub-turn-sim-model")
            self.assertEqual(
                interaction["turns"][0]["human_input"],
                "I am trying to think this through and want help comparing the tradeoffs.",
            )
            self.assertEqual(
                interaction["turns"][1]["ai_response"],
                "I can compare those factors and show where each option wins or loses.",
            )

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
                    return json.dumps({"human_input": "I want help comparing these two options before I decide."})
                return json.dumps(
                    {"ai_response": "I can compare them step by step and then give you a recommendation if you want one."}
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            single_scenario_pack = _write_single_scenario_pack(tmp_dir=tmp_dir, scenario_pack=scenario_pack)
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(FlakyTurnSimProvider(), "stub-flaky-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-turn-retry",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=single_scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm_turn_simulated",
                    enable_quality_gate=False,
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                interaction["turns"][0]["human_input"],
                "I want help comparing these two options before I decide.",
            )

    def test_llm_turn_simulation_salvages_truncated_json(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"

        class TruncatedTurnSimProvider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return '{\n  "human_input": "I want help thinking this through before I make the call."'
                if "human side" in system_prompt:
                    return json.dumps({"human_input": "Can you compare the main tradeoffs for me?"})
                return json.dumps(
                    {"ai_response": "Yes. I can compare the tradeoffs and point out which option fits your priorities better."}
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            single_scenario_pack = _write_single_scenario_pack(tmp_dir=tmp_dir, scenario_pack=scenario_pack)
            with patch(
                "src.dataset.generate_dataset.create_env_backed_provider",
                return_value=(TruncatedTurnSimProvider(), "stub-truncated-model"),
            ):
                dataset_root = generate_dataset(
                    dataset_name="synthetic-general-turn-truncated",
                    dataset_version="v2",
                    output_root=Path(tmp_dir),
                    scenario_pack_path=single_scenario_pack,
                    count_per_scenario=1,
                    seed=11,
                    generation_mode="llm_turn_simulated",
                    enable_quality_gate=False,
                )
            manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            first_item = manifest["items"][0]
            interaction = json.loads((dataset_root / first_item["relative_path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                interaction["turns"][0]["human_input"],
                "I want help thinking this through before I make the call.",
            )

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
                    payload = {
                        "human_input": [
                            "I need help comparing the options before I decide.",
                            "Can you compare the two biggest tradeoffs for me?",
                            "If you had to recommend one option, which would you pick?",
                        ][(self.call_count - 2) // 2 % 3]
                    }
                    return json.dumps(payload)
                payload = {
                    "ai_response": [
                        "Yes. Tell me which priorities matter most and I can help structure the comparison.",
                        "I can compare those tradeoffs and show where each option wins or loses.",
                        "I would recommend the option that best protects your priorities while keeping the upside.",
                    ][(self.call_count - 2) // 2 % 3]
                }
                return json.dumps(payload)

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
                            enable_quality_gate=False,
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
                        enable_quality_gate=False,
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

    def test_dataset_generation_rejects_low_quality_items(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"

        class LowQualityProvider:
            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                if "human side" in system_prompt:
                    return json.dumps({"human_input": "As an AI, I think you should decide."})
                return json.dumps({"ai_response": "As a language model, I recommend the first option."})

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "catalog.sqlite3"
            previous = os.environ.get("JDVP_CATALOG_DB_PATH")
            try:
                os.environ["JDVP_CATALOG_DB_PATH"] = str(db_path)
                with patch(
                    "src.dataset.generate_dataset.create_env_backed_provider",
                    return_value=(LowQualityProvider(), "low-quality-model"),
                ):
                    with self.assertRaises(RuntimeError):
                        generate_dataset(
                            dataset_name="synthetic-general-rejected",
                            dataset_version="v2",
                            output_root=Path(tmp_dir),
                            scenario_pack_path=scenario_pack,
                            count_per_scenario=1,
                            seed=11,
                            generation_mode="llm_turn_simulated",
                        )
                progress = json.loads((Path(tmp_dir) / "synthetic-general-rejected" / "v2" / "generation_progress.json").read_text(encoding="utf-8"))
                self.assertFalse(progress["is_complete"])
                from src.catalog.sqlite_store import CatalogStore

                generation_run_id = (
                    f"{(Path(tmp_dir) / 'synthetic-general-rejected' / 'v2').resolve(strict=False)}"
                    "::llm_turn_simulated::seed=11::count=1"
                )
                failed_items = CatalogStore(db_path).list_failed_dataset_generation_items(
                    generation_run_id=generation_run_id
                )
                self.assertTrue(any(row["status"] == "rejected" for row in failed_items))
            finally:
                if previous is None:
                    os.environ.pop("JDVP_CATALOG_DB_PATH", None)
                else:
                    os.environ["JDVP_CATALOG_DB_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
