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
