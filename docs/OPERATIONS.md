# POCv3 Operations

## Initial Operating Model

POCv3 starts with a single-run local workflow.

Target command shape:

```bash
python3 -m src.pipeline.run_poc \
  --input "data/raw/sample_interaction.json" \
  --track "heuristic_baseline" \
  --run-id "smoke-001"
```

Expected outputs:

- `data/runs/<run_id>/canonical/jsv/*.json`
- `data/runs/<run_id>/canonical/dv/*.json`
- `data/runs/<run_id>/canonical/trajectory.json`
- `data/runs/<run_id>/overlays/track_outputs.jsonl`
- `data/runs/<run_id>/manifest.json`

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
