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
- generated interactions stay raw-input compatible and can flow directly into `src.pipeline.run_poc`

Current M7 note:

- few-shot example packs are built from stored run extracts, not handwritten inline prompt blobs
- `src.dataset.build_fewshot_pack` materializes reusable example packs for observer research
- `fewshot_prompt` reuses the LLM observer adapter and injects serialized examples into the prompt
- `src.eval.fewshot_benchmark` turns dataset splits into executable few-shot evaluation plans
- benchmark slices remain the validation target for future retrieval or learned observers

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
