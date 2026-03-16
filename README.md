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
