# JDVP PoC v3

Proof-of-concept workspace for implementing JDVP against the current canonical protocol surface in `/Users/sangwon0001/Projects/bufferline/JDVP-protocol`.

## Intent

POCv3 is a core-first implementation workspace.

It separates:

- `protocol core`: canonical JSV/DV/Trajectory generation and schema validation
- `method layer`: how JSVs are extracted from interaction evidence
- `pipeline layer`: raw turns -> extraction -> protocol artifacts -> reports

## Scope

Initial scope is deliberately narrow:

- one human and one AI
- one interaction at a time
- `general` context first
- machine-facing JSON artifacts first
- descriptive outputs only

## Current Status

Current workspace status as of 2026-03-16:

- protocol-core generation and schema validation are implemented and covered by tests
- extraction tracks are available for `fixture_hint`, `heuristic_baseline`, `llm_observer`, and `fewshot_prompt`
- run storage persists canonical outputs, per-turn extracts, manifests, and resumable checkpoints
- benchmark and few-shot regression flows are implemented and used by the validation suite
- dataset generation supports stable `v1` regression packs plus richer `v2` research packs with turn variants and mixed blueprints
- preview generation is available for quick human review of richer datasets
- a lightweight SQLite catalog tracks generated datasets plus JDVP run state for recovery and reruns
- failed runs can be listed and retried by catalog state or scenario filter
- CI now uses the same validation entrypoint as local development

Current focus:

- keep `v1` stable as the regression baseline
- use `v2` and preview flows to expand research coverage without destabilizing CI
- improve operational recovery for unreliable LLM runs before adding heavier orchestration

## Design Rule

POCv3 follows JDVP `v1.4` exactly for protocol artifacts.

- JSV output must align to canonical `v1/schemas/jsv-schema.json`
- DV output must use ordinal derivation, not arbitrary transition constants
- `delta_judgment_holder` becomes `null` when a transition includes `Undefined`
- Trajectories contain adjacent-turn DVs only

## Repository Layout

```text
JDVP-POCv3/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MIGRATION_MATRIX.md
│   ├── MIGRATION_NOTES.md
│   └── OPERATIONS.md
├── config/
│   └── prompts/
├── data/
│   ├── fixtures/
│   ├── raw/
│   └── runs/
├── src/
│   ├── contracts/
│   ├── method/
│   ├── pipeline/
│   └── protocol_core/
└── tests/
```

## Build Order

1. protocol core
2. schema-aligned fixtures and tests
3. heuristic baseline track
4. one LLM extraction track
5. run storage and checkpointed result persistence
6. ensemble and benchmark evaluation
7. dataset generation and few-shot preparation
8. external-service modularization

## Roadmap

The detailed roadmap and milestone status live in [docs/IMPLEMENTATION_PLAN.md](/Users/sangwon0001/Projects/bufferline/JDVP-POCv3/docs/IMPLEMENTATION_PLAN.md).

## Source Dependency

POCv3 depends on the protocol repository as the semantic source of truth:

- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/protocol.md`
- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/docs/whitepaper.md`
- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/`

For local and CI validation, this repository keeps a vendored schema snapshot under `vendor/JDVP-protocol/v1/schemas/`.
Protocol semantics still originate in the canonical protocol repository and must be resynced from there when the spec changes.

Local schema sync check:

```bash
python3 scripts/check_protocol_schema_sync.py --require-upstream
```

Refresh vendored schema snapshot:

```bash
python3 scripts/sync_protocol_schema_snapshot.py
```

The refresh command also updates `vendor/JDVP-protocol/v1/schema_snapshot.json` so CI can verify vendored file integrity without requiring the sibling protocol repository.

## M1 Smoke Command

```bash
python3 -m src.pipeline.run_poc \
  --input data/fixtures/sample_interaction.json \
  --run-id smoke-local \
  --track fixture_hint
```

Expected outputs:

- `data/runs/<run_id>/canonical/jsv/`
- `data/runs/<run_id>/canonical/dv/`
- `data/runs/<run_id>/canonical/trajectory.json`
- `data/runs/<run_id>/overlays/track_outputs.jsonl`
- `data/runs/<run_id>/manifest.json`

M2 baseline command:

```bash
python3 -m src.pipeline.run_poc \
  --input data/fixtures/sample_interaction.json \
  --run-id baseline-local \
  --track heuristic_baseline
```

M3 local LLM command:

```bash
export JDVP_LLM_BASE_URL=http://localhost:11434/v1
export JDVP_LLM_API_KEY=dummy
export JDVP_LLM_MODEL=llama3.2

python3 -m src.pipeline.run_poc \
  --input data/fixtures/sample_interaction.json \
  --run-id llm-local \
  --track llm_observer
```

CI does not call real providers. Real LLM execution is for local research runs only.

M4 resumable run command:

```bash
python3 -m src.pipeline.run_poc \
  --input data/fixtures/sample_interaction.json \
  --run-id baseline-local \
  --track heuristic_baseline \
  --resume
```

Additional run outputs now include:

- `data/runs/<run_id>/extracts/<track_name>/`
- `data/runs/<run_id>/checkpoints/progress.json`

M5 benchmark command:

```bash
python3 -m src.eval.ensemble_benchmark \
  --run-dir data/runs/fixture-run \
  --run-dir data/runs/baseline-run \
  --output-dir data/runs/benchmark-report
```

M6 dataset generation command:

```bash
python3 -m src.dataset.generate_dataset \
  --dataset-name synthetic-general \
  --dataset-version v1 \
  --scenario-pack config/datasets/general_scenarios_v1.json \
  --count-per-scenario 2 \
  --seed 11
```

Expected dataset outputs:

- `data/generated/<dataset_name>/<dataset_version>/interactions/*.json`
- `data/generated/<dataset_name>/<dataset_version>/manifest.json`
- `data/generated/<dataset_name>/<dataset_version>/splits.json`

Dataset-scoped JDVP run:

```bash
python3 scripts/run_dataset.py \
  --dataset-root data/generated/synthetic-general/v1 \
  --output-root data/runs/dataset-fixture \
  --track fixture_hint \
  --split test
```

This writes `dataset_run_summary.json` plus one run directory per selected interaction.

Model-by-model dataset run matrix:

```bash
python3 scripts/run_dataset_matrix.py --config path/to/matrix-config.json
```

The config should point at one dataset slice and a list of track/model environment setups. The script writes `dataset_matrix_summary.json` so repeated model runs can be compared on the same slice.

Benchmark a dataset matrix against one baseline label:

```bash
python3 scripts/benchmark_dataset_matrix.py \
  --matrix-summary path/to/dataset_matrix_summary.json \
  --baseline-label fixture \
  --output-root data/runs/matrix-benchmark
```

This writes pairwise comparison reports plus interaction-level ensemble reports for the completed runs in the matrix.

Richer research pack with mixed blueprints:

```bash
python3 -m src.dataset.generate_dataset \
  --dataset-name synthetic-general-rich \
  --dataset-version v1 \
  --scenario-pack config/datasets/general_scenarios_v2.json \
  --count-per-scenario 4 \
  --seed 11
```

Quick preview for human review:

```bash
python3 scripts/preview_dataset.py
```

This writes `preview.json` next to the generated preview dataset so you can skim turn text, blueprint choice, and JSV hints without running the full benchmark loop.

Operational catalog:

- POCv3 now keeps a lightweight SQLite catalog at `data/catalog/pocv3.sqlite3` by default.
- Set `JDVP_CATALOG_DB_PATH` to redirect the catalog location for local experiments or CI.
- The catalog indexes generated datasets plus JDVP run status so failed LLM runs can be recovered without scanning the filesystem.

Retry failed runs:

```bash
python3 scripts/rerun_failed_runs.py --limit 20
python3 scripts/rerun_failed_runs.py --dataset-id generated/synthetic-general/v1 --limit 20
python3 scripts/rerun_failed_runs.py --scenario-id travel-planning --limit 10
```

This replays failed cataloged runs using the original `input_path` and `track_name`. By default it creates new run ids with a `-retry-XX` suffix so previous failed artifacts remain inspectable.

Inspect failed runs:

```bash
python3 scripts/list_failed_runs.py
python3 scripts/list_failed_runs.py --dataset-id generated/synthetic-general/v1 --limit 10
python3 scripts/list_failed_runs.py --scenario-id travel-planning --limit 10
python3 scripts/list_failed_runs.py --summary-by-scenario
```

Use this before reruns to see which scenarios or inputs are failing most often.

M7 few-shot pack command:

```bash
python3 -m src.dataset.build_fewshot_pack \
  --run-dir data/runs/fixture-run \
  --output data/fewshot/general-fixture-pack-v1.json \
  --max-examples 3
```

M7 few-shot observer command:

```bash
export JDVP_LLM_BASE_URL=http://localhost:11434/v1
export JDVP_LLM_API_KEY=dummy
export JDVP_LLM_MODEL=llama3.2
export JDVP_FEWSHOT_PACK_PATH=data/fewshot/general-fixture-pack-v1.json

python3 -m src.pipeline.run_poc \
  --input data/generated/synthetic-general/v1/interactions/synthetic-general-job-offer-000.json \
  --run-id fewshot-local \
  --track fewshot_prompt
```

M7 benchmark plan command:

```bash
python3 -m src.eval.fewshot_benchmark \
  --dataset-root data/generated/synthetic-general/v1 \
  --fewshot-pack data/fewshot/general-fixture-pack-v1.json \
  --split test \
  --output data/runs/fewshot-test-plan.json \
  --max-examples 3
```

M7 benchmark runner command:

```bash
export JDVP_LLM_BASE_URL=http://localhost:11434/v1
export JDVP_LLM_API_KEY=dummy
export JDVP_LLM_MODEL=llama3.2

python3 -m src.eval.run_fewshot_benchmark \
  --plan data/runs/fewshot-test-plan.json \
  --output-root data/runs/fewshot-test-results
```

M7 regression suite command:

```bash
python3 scripts/run_fewshot_regression_suite.py \
  --config config/eval/fewshot_regression_general_v1.json \
  --output-root data/baselines/fewshot_regression_general_v1 \
  --clean
```

The checked-in deterministic baseline lives under `data/baselines/fewshot_regression_general_v1/`.

Standard local validation bundle:

```bash
python3 scripts/run_validation_suite.py
```

This writes a step-by-step summary to `data/validation/latest/validation_summary.json`.
CI uses the same entrypoint with `--allow-missing-upstream` so local and CI validation stay aligned.

M8 service-facing Python entrypoint:

```python
from pathlib import Path

from src.service import RunRequest, run_interaction

result = run_interaction(
    RunRequest(
        input_path=Path("data/fixtures/sample_interaction.json"),
        run_id="service-local",
        output_root=Path("data/runs"),
        track_name="fixture_hint",
    )
)

print(result.run_dir)
```

Service responses can also be serialized as stable envelopes:

```python
from pathlib import Path

from src.service import RunRequest, run_interaction_response

payload = run_interaction_response(
    RunRequest(
        input_path=Path("data/fixtures/sample_interaction.json"),
        run_id="service-local",
        output_root=Path("data/runs"),
    )
)

print(payload["ok"])
print(payload["schema_version"])
```

M8 JSON CLI adapter:

```bash
cat > /tmp/jdvp-run.json <<'JSON'
{
  "operation": "run_interaction",
  "input_path": "data/fixtures/sample_interaction.json",
  "run_id": "json-cli-local",
  "output_root": "data/runs",
  "track_name": "fixture_hint"
}
JSON

python3 -m src.service.json_api --input /tmp/jdvp-run.json
```
