# POCv3 Operations

## Initial Operating Model

POCv3 starts with a single-run local workflow.

Target command shape:

```bash
python3 -m src.pipeline.run_poc \
  --input "data/raw/sample_interaction.json" \
  --track "fixture_hint" \
  --run-id "smoke-001"
```

Expected outputs:

- `data/runs/<run_id>/canonical/jsv/*.json`
- `data/runs/<run_id>/canonical/dv/*.json`
- `data/runs/<run_id>/canonical/trajectory.json`
- `data/runs/<run_id>/manifest.json`

Current M1 note:

- the pipeline uses fixture-provided `meta.jsv_hint` values to exercise canonical protocol generation
- extraction tracks are added in M2

Current M2 note:

- `heuristic_baseline` is now available as the first method-layer track
- canonical JSV, DV, and Trajectory artifacts are still derived in the protocol core, not inside the track

Current M3 note:

- `llm_observer` uses provider-backed extraction for local research runs
- CI validates normalization and adapter behavior with fake provider responses, not live network calls

Current M4 note:

- runs now persist per-turn extracts under `extracts/<track_name>/`
- progress snapshots are written to `checkpoints/progress.json`
- `--resume` reuses stored turn extracts for the same `run_id`

Current M5 note:

- benchmark runs compare stored extracts from multiple runs of the same interaction
- output includes `benchmark_summary.json`, `turn_comparisons.jsonl`, and an ensemble manifest
- simple ensemble output is derived from majority agreement across track JSV hints

Current M6 note:

- synthetic datasets are generated under `data/generated/<dataset_name>/<dataset_version>/`
- each dataset writes a versioned `manifest.json` plus `splits.json`
- scenario packs live under `config/datasets/` and drive reproducible synthetic interaction generation
- `python3 -m src.dataset.generate_dataset --generation-mode llm` uses the configured JDVP LLM provider to rewrite turn utterances for research datasets while preserving scenario structure and `meta.jsv_hint`
- `python3 -m src.dataset.generate_dataset --generation-mode llm_turn_simulated` runs a stateful turn-by-turn human/assistant simulation while keeping the selected scenario blueprint and hidden JDVP targets
- LLM-backed generation now tracks accepted, failed, and rejected items in SQLite so reruns only retry incomplete items
- `--disable-quality-gate` is available for debugging, but research runs should keep the quality gate enabled
- generated interactions stay raw-input compatible and can flow directly into `src.pipeline.run_poc`

Current M7 note:

- few-shot example packs are built from stored run extracts, not handwritten inline prompt blobs
- `src.dataset.build_fewshot_pack` materializes reusable example packs for observer research
- `fewshot_prompt` reuses the LLM observer adapter and injects serialized examples into the prompt
- `src.eval.fewshot_benchmark` turns dataset splits into executable few-shot evaluation plans
- `src.eval.run_fewshot_benchmark` executes those plans and writes per-interaction comparison reports
- few-shot benchmark results now include per-field disagreement rates plus scenario and context-module breakdowns
- benchmark execution can fail on configured disagreement thresholds while still writing `benchmark_results.json`
- `scripts/run_fewshot_regression_suite.py` is the standard end-to-end benchmark entrypoint for this milestone
- `config/eval/fewshot_regression_general_v1.json` defines the checked-in deterministic regression policy and thresholds
- `data/baselines/fewshot_regression_general_v1/` holds the current deterministic baseline artifacts
- benchmark slices remain the validation target for future retrieval or learned observers

Current M8 note:

- `src.service` is the official reusable boundary for external consumers
- `src.service.json_api` is the official transport adapter for this milestone
- transport responses are versioned and serialized as success/error envelopes
- HTTP transport is intentionally deferred until a real consumer needs it

Current operational note:

- `scripts/run_validation_suite.py` is the standard local and CI validation entrypoint
- `data/catalog/pocv3.sqlite3` is the default lightweight catalog for dataset and run tracking
- `scripts/list_failed_runs.py` and `scripts/rerun_failed_runs.py` are the current recovery tools for unreliable LLM runs
- `scripts/run_dataset.py` now registers one cataloged `dataset run` unit per batch output root, with aggregate status and counts
- `scripts/list_failed_runs.py --dataset-runs` lists dataset-level recovery units in addition to per-run failures
- `scripts/rerun_failed_runs.py --dataset-run-id <id>` replays the failed subset inside one cataloged dataset run
- `scripts/list_generation_runs.py` lists dataset-generation runs and their partial/completed states
- `scripts/list_failed_generation_items.py --generation-run-id <id>` shows which generated items failed and will be retried
- `scripts/rerun_failed_generation_items.py --generation-run-id <id>` retries failed generation items for a given generation run
- rerun the same `python3 -m src.dataset.generate_dataset ...` command to retry only failed generation items for that dataset root and generation mode
- `scripts/validate_contracts.py` validates service contract DTOs against the vendored schema
- `scripts/prepare_remote_turn_sim_trial100_matrix.py` prepares configuration for 100-item turn-simulated generation trials across remote model targets
- the immediate next operational checkpoint is a local 100-item `llm_turn_simulated` run to tune model choice and rejection thresholds
- `config/datasets/general_scenarios_v1.json` remains the stable regression pack
- `config/datasets/general_scenarios_v2.json` is the richer research pack and should be reviewed through preview flows before it influences baseline policy

## Current Research Commands

Use these commands to reproduce the current `5-turn` generation experiments.

Remote `gpt-oss:20b` on `192.168.50.201`:

```bash
export JDVP_LLM_BASE_URL=http://192.168.50.201:11434/v1
export JDVP_LLM_API_KEY=dummy
export JDVP_LLM_MODEL=gpt-oss:20b
export JDVP_LLM_TIMEOUT_SECONDS=240

python3 -m src.dataset.generate_dataset \
  --dataset-name remote-5turn-gptoss20b-300 \
  --dataset-version v2 \
  --output-root data/generated-5turn-runs \
  --scenario-pack config/datasets/general_scenarios_5turn_smoke.json \
  --count-per-scenario 150 \
  --generation-mode llm_turn_simulated
```

Local `gemma3:27b`:

```bash
export JDVP_LLM_BASE_URL=http://localhost:11434/v1
export JDVP_LLM_API_KEY=dummy
export JDVP_LLM_MODEL=gemma3:27b
export JDVP_LLM_TIMEOUT_SECONDS=240

python3 -m src.dataset.generate_dataset \
  --dataset-name local-5turn-gemma27b-300 \
  --dataset-version v2 \
  --output-root data/generated-5turn-runs \
  --scenario-pack config/datasets/general_scenarios_5turn_smoke.json \
  --count-per-scenario 150 \
  --generation-mode llm_turn_simulated
```

Remote `gemma3:4b` on `192.168.50.132`:

```bash
export JDVP_LLM_BASE_URL=http://192.168.50.132:11434/v1
export JDVP_LLM_API_KEY=dummy
export JDVP_LLM_MODEL=gemma3:4b
export JDVP_LLM_TIMEOUT_SECONDS=240

python3 -m src.dataset.generate_dataset \
  --dataset-name remote-5turn-gemma4b-300 \
  --dataset-version v2 \
  --output-root data/generated-5turn-runs \
  --scenario-pack config/datasets/general_scenarios_5turn_smoke.json \
  --count-per-scenario 150 \
  --generation-mode llm_turn_simulated
```

Progress checks:

```bash
find data/generated-5turn-runs/remote-5turn-gptoss20b-300/v2/interactions -name '*.json' | wc -l
find data/generated-5turn-runs/local-5turn-gemma27b-300/v2/interactions -name '*.json' | wc -l
find data/generated-5turn-runs/remote-5turn-gemma4b-300/v2/interactions -name '*.json' | wc -l
```

Current completed datasets:

- `data/generated-5turn-runs/remote-5turn-gptoss20b-300/v2`
- `data/generated-5turn-runs/local-5turn-gemma27b-300/v2`
- `data/generated-5turn-runs/remote-5turn-gemma4b-300/v2`

Manual review summary:

- `gpt-oss:20b` produced the most natural `5-turn` conversations and the clearest gradual state change
- `gemma3:27b` was slightly more rigid but still strong enough as the local high-quality fallback
- `gemma3:4b` was usable for research-scale generation, but noticeably more generic and repetitive than the other two

Schema sync check:

- run `python3 scripts/check_protocol_schema_sync.py --require-upstream` before or alongside vendor snapshot refreshes
- the command compares the sibling `JDVP-protocol/v1/schemas/` files against `vendor/JDVP-protocol/v1/schemas/`
- without `--require-upstream`, the command exits cleanly when the sibling protocol repository is not present locally
- refresh the vendored snapshot with `python3 scripts/sync_protocol_schema_snapshot.py`
- the refresh command copies the three canonical schema files and then rechecks for drift
- refresh also rewrites `vendor/JDVP-protocol/v1/schema_snapshot.json` with upstream revision and vendored file hashes
- CI validates that manifest against the vendored files even when the sibling protocol repository is absent

## Validation Requirements

Every run must verify:

1. raw input contract
2. canonical JSV schema compliance
3. canonical DV schema compliance
4. canonical trajectory schema compliance
5. adjacent-turn continuity inside the trajectory

## Run Modes

### Smoke

Use one local fixture and one deterministic extractor.

Purpose:

- verify pipeline wiring
- verify schema alignment
- verify run artifact layout

### Baseline

Use heuristic extraction on a small fixture batch.

Purpose:

- produce a stable regression floor
- catch protocol-core regressions before LLM variability is introduced

### LLM Observer

Use one provider-backed extractor after the baseline path is stable.

Purpose:

- compare method-layer outputs without changing core artifacts

## Non-Goals For Initial Operations

- scheduled jobs
- budget routers
- promotion gates
- dataset generation farms
- registry bookkeeping

Those may return later, but they are intentionally absent from initial POCv3 operations.
