from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset.build_fewshot_pack import build_fewshot_pack
from src.dataset.generate_dataset import generate_dataset
from src.eval.fewshot_benchmark import build_fewshot_benchmark_plan
from src.service import (
    FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION,
    RUN_RESULT_SCHEMA_VERSION,
    SERVICE_ERROR_SCHEMA_VERSION,
    SERVICE_RESPONSE_SCHEMA_VERSION,
    FewshotBenchmarkRequest,
    RunRequest,
    run_fewshot_benchmark,
    run_fewshot_benchmark_response,
    run_interaction,
    run_interaction_file,
    run_interaction_response,
)
from src.service.poc_service import build_pipeline_artifacts


ROOT = Path(__file__).resolve().parents[1]
VALID_RESPONSE = """
{
  "judgment_holder": "AI",
  "delegation_awareness": "Implicit",
  "cognitive_engagement": "Reactive",
  "information_seeking": "Passive",
  "confidence": {
    "judgment_holder": "medium",
    "delegation_awareness": "medium",
    "cognitive_engagement": "medium",
    "information_seeking": "medium"
  },
  "evidence_spans": [
    {"text": "Just recommend one plan for me.", "category": "delegation_signal"}
  ],
  "observer_confidence": 0.77,
  "observer_notes": "Detected direct delegation request"
}
"""


class ServiceApiTests(unittest.TestCase):
    def test_build_pipeline_artifacts_returns_canonical_sequences(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        raw_interaction = json.loads(fixture.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = build_pipeline_artifacts(
                raw_interaction=raw_interaction,
                track_name="fixture_hint",
                run_dir=Path(tmp_dir) / "service-run",
            )
            self.assertEqual(len(artifacts.jsv_sequence), 3)
            self.assertEqual(len(artifacts.dv_sequence), 2)
            self.assertEqual(artifacts.trajectory["interaction_id"], "fixture-general-001")

    def test_run_interaction_file_writes_outputs(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_interaction_file(
                input_path=fixture,
                run_id="service-run",
                output_root=Path(tmp_dir),
                track_name="fixture_hint",
            )
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "canonical" / "trajectory.json").is_file())

    def test_run_interaction_returns_typed_result(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_interaction(
                RunRequest(
                    input_path=fixture,
                    run_id="service-run",
                    output_root=Path(tmp_dir),
                    track_name="fixture_hint",
                )
            )
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.trajectory_path.is_file())
            self.assertTrue(result.extracts_path.is_file())
            self.assertIn("run_dir", result.to_dict())
            external = result.to_external_dict()
            self.assertEqual(external["schema_version"], RUN_RESULT_SCHEMA_VERSION)
            self.assertNotIn("extracts_path", external)

    def test_run_fewshot_benchmark_returns_typed_result(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=output_root,
                scenario_pack_path=scenario_pack,
                count_per_scenario=2,
                seed=11,
            )
            seed_run_dir = run_interaction_file(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root / "seed-runs",
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=seed_run_dir, output_path=pack_path, max_examples=3)
            plan_path = output_root / "fewshot-test-plan.json"
            build_fewshot_benchmark_plan(
                dataset_root=dataset_root,
                fewshot_pack_path=pack_path,
                split="test",
                output_path=plan_path,
                max_examples=2,
            )
            with patch.dict(
                os.environ,
                {
                    "JDVP_LLM_BASE_URL": "http://localhost:11434/v1",
                    "JDVP_LLM_API_KEY": "dummy",
                    "JDVP_LLM_MODEL": "fake-model",
                },
                clear=False,
            ):
                with patch(
                    "src.method.tracks.llm_observer.OpenAICompatibleProvider.generate",
                    return_value=VALID_RESPONSE,
                ):
                    result = run_fewshot_benchmark(
                        FewshotBenchmarkRequest(
                            plan_path=plan_path,
                            output_root=output_root / "benchmark-results",
                        )
                    )
            self.assertTrue(result.results_path.is_file())
            self.assertIn("results_path", result.to_dict())
            external = result.to_external_dict()
            self.assertEqual(external["schema_version"], FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION)
            self.assertEqual(external["item_count"], 2)

    def test_run_interaction_response_serializes_success(self) -> None:
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = run_interaction_response(
                RunRequest(
                    input_path=fixture,
                    run_id="service-run",
                    output_root=Path(tmp_dir),
                    track_name="fixture_hint",
                )
            )
            self.assertEqual(response["schema_version"], SERVICE_RESPONSE_SCHEMA_VERSION)
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["schema_version"], RUN_RESULT_SCHEMA_VERSION)
            self.assertIn("trajectory_path", response["result"])
            self.assertNotIn("extracts_path", response["result"])

    def test_run_interaction_response_serializes_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = run_interaction_response(
                RunRequest(
                    input_path=Path(tmp_dir) / "missing.json",
                    run_id="service-run",
                    output_root=Path(tmp_dir),
                    track_name="fixture_hint",
                )
            )
            self.assertEqual(response["schema_version"], SERVICE_RESPONSE_SCHEMA_VERSION)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "input_not_found")
            self.assertEqual(response["error"]["schema_version"], SERVICE_ERROR_SCHEMA_VERSION)

    def test_run_fewshot_benchmark_response_serializes_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = run_fewshot_benchmark_response(
                FewshotBenchmarkRequest(
                    plan_path=Path(tmp_dir) / "missing-plan.json",
                    output_root=Path(tmp_dir) / "benchmark-results",
                )
            )
            self.assertEqual(response["schema_version"], SERVICE_RESPONSE_SCHEMA_VERSION)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "benchmark_plan_not_found")
            self.assertEqual(response["error"]["schema_version"], SERVICE_ERROR_SCHEMA_VERSION)

    def test_run_fewshot_benchmark_response_serializes_success_contract(self) -> None:
        scenario_pack = ROOT / "config" / "datasets" / "general_scenarios_v1.json"
        fixture = ROOT / "data" / "fixtures" / "sample_interaction.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            dataset_root = generate_dataset(
                dataset_name="synthetic-general",
                dataset_version="v1",
                output_root=output_root,
                scenario_pack_path=scenario_pack,
                count_per_scenario=2,
                seed=11,
            )
            seed_run_dir = run_interaction_file(
                input_path=fixture,
                run_id="fixture-pack",
                output_root=output_root / "seed-runs",
                track_name="fixture_hint",
            )
            pack_path = output_root / "fewshot-pack.json"
            build_fewshot_pack(run_dir=seed_run_dir, output_path=pack_path, max_examples=3)
            plan_path = output_root / "fewshot-test-plan.json"
            build_fewshot_benchmark_plan(
                dataset_root=dataset_root,
                fewshot_pack_path=pack_path,
                split="test",
                output_path=plan_path,
                max_examples=2,
            )
            with patch.dict(
                os.environ,
                {
                    "JDVP_LLM_BASE_URL": "http://localhost:11434/v1",
                    "JDVP_LLM_API_KEY": "dummy",
                    "JDVP_LLM_MODEL": "fake-model",
                },
                clear=False,
            ):
                with patch(
                    "src.method.tracks.llm_observer.OpenAICompatibleProvider.generate",
                    return_value=VALID_RESPONSE,
                ):
                    response = run_fewshot_benchmark_response(
                        FewshotBenchmarkRequest(
                            plan_path=plan_path,
                            output_root=output_root / "benchmark-results",
                        )
                    )
            self.assertEqual(response["schema_version"], SERVICE_RESPONSE_SCHEMA_VERSION)
            self.assertTrue(response["ok"])
            self.assertEqual(
                response["result"]["schema_version"],
                FEWSHOT_BENCHMARK_RESULT_SCHEMA_VERSION,
            )
            self.assertEqual(response["result"]["item_count"], 2)


if __name__ == "__main__":
    unittest.main()
