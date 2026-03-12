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
- dataset governance
- silver/gold promotion

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

## Contracts

### Canonical Protocol Contract

POCv3 does not define its own JSV/DV/Trajectory schema variants for core artifacts.

It uses:

- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/schemas/jsv-schema.json`
- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/schemas/dv-schema.json`
- `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/schemas/trajectory-schema.json`

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

### M4: Comparative Evaluation

- compare tracks on same fixtures
- disagreement summary as overlay output

### M5: Research Automation

- only if needed
- silver/gold promotion and release gating can return after the core path is stable

## Explicit Omissions For Initial Build

- web dashboard
- silver/gold governance loops
- registry systems
- storage backend abstraction
- benchmark farms
- productized reporting surfaces
