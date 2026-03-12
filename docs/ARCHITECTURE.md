# POCv3 Architecture

## Goal

Build a minimal but correct implementation path for JDVP `v1.4`.

The first successful path is:

`raw turns -> JSV sequence -> DV sequence -> Trajectory -> schema validation`

## Architecture Layers

### 1. Protocol Core

Purpose:

- hold canonical JDVP artifact logic
- remain independent from extraction strategy
- stay close to the protocol repo's current semantics

Modules:

- `src/protocol_core/jsv_types.py`
- `src/protocol_core/dv_ordinal.py`
- `src/protocol_core/trajectory.py`
- `src/protocol_core/schema_validate.py`

Responsibilities:

- represent canonical JSV fields
- derive DV values from ordinal state positions
- apply `Undefined -> null` handling for `delta_judgment_holder`
- enforce adjacent-turn continuity for trajectories
- validate generated artifacts against canonical schemas in `JDVP-protocol/v1/schemas/`

Non-responsibilities:

- LLM prompting
- heuristic extraction
- dataset governance policy
- benchmark orchestration
- service integration

### 2. Method Layer

Purpose:

- convert observed turn evidence into candidate JSV assignments
- keep extraction-specific metadata outside canonical protocol fields

Modules:

- `src/method/tracks/`
- `src/method/evidence/`
- `src/method/normalization/`

Expected output:

- canonical `jsv`
- optional overlay fields such as `evidence_spans`, `observer_notes`, `track_id`, `model_id`

Track order:

1. `heuristic_baseline`
2. `llm_observer_local`
3. `llm_observer_remote`
4. `fewshot_retrieval_track`

### 3. Pipeline Layer

Purpose:

- orchestrate one end-to-end run
- produce stable run artifacts
- keep operational state outside protocol core

Module:

- `src/pipeline/run_poc.py`

Responsibilities:

- load raw interaction turns
- call one extraction track
- build JSV/DV/Trajectory
- run schema validation
- write outputs under `data/runs/<run_id>/`

Future expansion:

- checkpoint intermediate results
- persist per-track extracts separately from canonical artifacts
- support ensemble and benchmark passes over stored run outputs

Current M4 shape:

- `extracts/<track_name>/turn-<n>.json` stores per-turn intermediate extract state
- `extracts/<track_name>/extracts.jsonl` stores batch-oriented extract rows
- `checkpoints/progress.json` stores resumable run progress

Current M5 shape:

- `src/eval/ensemble_benchmark.py` compares stored extract rows across runs
- benchmark output is written separately from canonical protocol artifacts
- simple ensemble output is derived at the JSV-hint layer and does not alter canonical stored artifacts

## Contracts

### Canonical Protocol Contract

POCv3 does not define its own JSV/DV/Trajectory schema variants for core artifacts.

It uses:

- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/schemas/jsv-schema.json`
- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/schemas/dv-schema.json`
- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/schemas/trajectory-schema.json`

Operational note:

- POCv3 keeps a vendored schema snapshot in `vendor/JDVP-protocol/v1/schemas/` for CI and local portability
- when canonical protocol schemas change, the vendored snapshot must be refreshed in the same change set

### Overlay Contract

POCv3 may define non-canonical overlay records for extraction metadata.

Examples:

- `evidence_spans`
- `track_output`
- `run_manifest`
- `observer_confidence`

These records must never replace or mutate canonical protocol meaning.

## Data Flow

1. Load raw interaction record.
2. Normalize one interaction into ordered turns.
3. Extract JSV per human-observed turn.
4. Derive adjacent-turn DVs from JSVs.
5. Build one trajectory for one `interaction_id`.
6. Validate each artifact against canonical protocol schemas.
7. Write canonical artifacts and overlay artifacts separately.

## Directory Roles

### `data/raw/`

Input conversations not yet converted into protocol artifacts.

### `data/fixtures/`

Small stable examples for smoke tests and contract tests.

### `data/runs/`

Generated run outputs.

Suggested layout:

```text
data/runs/<run_id>/
├── input/
├── canonical/
│   ├── jsv/
│   ├── dv/
│   └── trajectory/
├── overlays/
└── manifest.json
```

Future expansion:

```text
data/runs/<run_id>/
├── checkpoints/
├── extracts/
├── ensembles/
├── reports/
└── benchmark/
```

## Milestones

### M1: Core Valid Path

- raw turn fixture loads
- JSV/DV/Trajectory generated
- canonical schema validation passes

### M2: Baseline Extractor

- deterministic heuristic extractor
- evidence span capture
- one smoke CLI command

### M3: LLM Observer

- local or remote track adapter
- strict JSON output normalization
- extraction failure fallback path
- local real-model execution path for direct research runs

### M4: Run Storage And Checkpointing

- persist per-track extracts and overlay outputs under versioned run directories
- add resumable intermediate state for long-running extraction jobs
- separate canonical artifacts from method-specific intermediate outputs

### M5: Ensemble And Benchmark

- compare tracks on the same fixture or dataset slice
- disagreement summary and simple ensemble output
- benchmark reports over stored extracts
- regression comparisons against heuristic baseline

### M6: Dataset And Generation Loop

- version raw, generated, and fixture datasets
- add synthetic chat generation and scenario packs
- prepare train/valid/test style splits for future few-shot work
- store provenance for generated conversations and extracted labels

### M7: Few-Shot And Learned Tracks

- build few-shot example packs from stored extracts
- support retrieval-based or trained observer tracks
- add validation flow for learned-track outputs against stable benchmarks

### M8: Service Modularization

- package reusable modules for external service use
- expose stable boundaries for extraction, protocol-core generation, and validation
- keep service-facing interfaces separate from research-only orchestration

## Explicit Omissions For Initial Build

- web dashboard
- silver/gold governance loops
- registry systems
- heavyweight storage backend abstraction
- benchmark farms before run storage exists
- productized reporting surfaces
