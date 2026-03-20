# POCv3 Implementation Plan

## Phase 0: Scaffold Freeze

Goal:

- freeze the repository shape and migration decisions before code import begins

Tasks:

1. approve migration matrix
2. confirm protocol repo path strategy
3. confirm Python packaging choice

Exit condition:

- no disagreement about `keep/adapt/drop`

## Phase 1: M1 Core Valid Path

Goal:

- produce canonical artifacts from one raw fixture

Tasks:

1. define raw interaction schema in `src/contracts/raw_interaction.schema.json`
2. create one fixture under `data/fixtures/`
3. implement canonical JSV type helpers
4. implement ordinal DV derivation
5. implement trajectory builder
6. implement canonical schema validator
7. add smoke CLI runner
8. add tests for canonical examples and fixture path

Exit condition:

- one command emits schema-valid JSV, DV, and Trajectory outputs

## Phase 2: M2 Heuristic Baseline

Goal:

- attach one deterministic extraction method without polluting protocol core

Tasks:

1. define track interface
2. port only useful baseline heuristics from POCv1/POCv2
3. emit overlay metadata with evidence spans
4. keep track metadata outside canonical JSV/DV payloads
5. add regression fixtures

Exit condition:

- baseline run is repeatable and produces stable artifacts

## Phase 3: M3 LLM Observer

Goal:

- add one provider-backed extractor behind the same track contract

Tasks:

1. define provider adapter boundary
2. add prompt pack layout in `config/prompts/`
3. normalize model responses into canonical JSV assignments
4. reject malformed model output before protocol artifact generation
5. compare LLM output against baseline on shared fixtures
6. keep real provider execution as a local research path, not a CI dependency

Exit condition:

- one LLM track runs through the same pipeline without special-casing protocol logic

## Phase 4: M4 Run Storage And Checkpointing

Goal:

- persist method outputs and resume longer research runs safely

Tasks:

1. add per-run extract storage separate from canonical outputs
2. add checkpoint/progress persistence for interrupted runs
3. add run manifests that capture track, prompt, dataset, and code revision inputs
4. keep canonical artifacts reproducible from stored intermediate results

Exit condition:

- long-running extraction jobs can resume without corrupting canonical outputs

## Phase 5: M5 Ensemble And Benchmark

Goal:

- compare and combine track outputs over stored runs

Tasks:

1. add disagreement report generation
2. add simple ensemble rules over multiple track outputs
3. add benchmark fixtures or dataset slices
4. add regression comparisons against the heuristic baseline

Exit condition:

- benchmark and ensemble outputs are available without changing protocol-core semantics

## Phase 6: M6 Dataset And Generation Loop

Goal:

- support durable research inputs rather than one-off smoke fixtures

Tasks:

1. add dataset manifests and version keys
2. add synthetic chat generation and scenario packs
3. add generated/raw/fixture dataset separation
4. add split definitions for future evaluation and few-shot work

Exit condition:

- datasets are versioned, reproducible, and reusable across runs

## Phase 7: M7 Few-Shot And Learned Tracks

Goal:

- prepare future trainable or retrieval-based observers

Tasks:

1. build few-shot example packs from vetted extracts
2. add retrieval-based or learned observer track entrypoints
3. validate learned-track outputs against benchmark slices
4. keep protocol-core logic independent from training logic

Exit condition:

- a few-shot or learned track can run under the same extraction contract

## Phase 8: M8 Service Modularization

Goal:

- make the system consumable by external services without exposing research internals

Tasks:

1. define stable module boundaries for protocol core, tracks, and validation
2. isolate service-facing APIs from research orchestration
3. document reusable integration surfaces

Exit condition:

- external systems can consume extraction and validation modules without importing the full research workspace

Current status:

- complete for the initial milestone
- official external interface is `src.service` plus the JSON adapter in `src.service.json_api`
- external contracts now use versioned DTOs and success/error envelopes
- HTTP transport remains deferred because the service contract is the stable boundary, not the transport choice

## Current Status Snapshot

Implemented and working:

- M1 protocol-core generation and canonical schema validation
- M2 deterministic heuristic baseline extraction
- M3 provider-backed LLM observer plus few-shot prompt track
- M4 run storage, per-turn extracts, manifests, and resumable checkpoints
- M5 disagreement reporting, ensemble comparison, and few-shot regression suite
- M6 dataset generation, versioned manifests, richer `v2` scenario packs, and preview generation
- provider-backed LLM utterance materialization for research dataset generation
- provider-backed turn-by-turn dataset simulation for richer research conversations
- dataset-generation SQLite recovery with accepted/failed item resume behavior
- lightweight quality gates for LLM-backed dataset generation with `rejected` item tracking
- M7 few-shot pack creation plus benchmark planning and execution
- M8 service boundary, JSON transport, and local/CI validation unification

Operational additions beyond the original milestone text:

- lightweight SQLite catalog for generated datasets and JDVP run state
- failed-run inspection and rerun scripts for unreliable LLM execution
- richer research dataset blueprints separated from the stable regression dataset
- dataset-generation run inspection and failed-item inspection scripts

## Roadmap

### Priority 1: Operational Recovery

Goal:

- reduce wasted time when provider-backed runs fail repeatedly

Tasks:

1. expand catalog queries for dataset-level and track-level failure inspection
2. connect rerun tooling to scenario, dataset, and status slices more cleanly
3. decide when catalog state should drive resume behavior automatically

Exit condition:

- failed LLM runs can be identified and replayed quickly without manual filesystem triage

### Priority 2: Dataset-Scoped Execution

Goal:

- move from single-run tooling to repeatable dataset-level JDVP execution

Tasks:

1. add a batch runner that executes JDVP across one dataset slice with one selected track/model
2. persist dataset-level execution metadata so one dataset can be rerun or resumed cleanly
3. connect catalog filters to dataset-level execution outputs instead of only single-run records

Exit condition:

- one command can run JDVP over a dataset slice and recover the failed subset without manual file selection

### Priority 3: Model Benchmarking And Ensemble Evaluation

Goal:

- compare tracks and models as experiment units rather than isolated runs

Tasks:

1. record model/provider/prompt metadata more consistently for provider-backed runs
2. add model-level benchmark summaries across dataset slices
3. support dataset-level comparison inputs for ensemble experiments
4. expose a lightweight leaderboard view for disagreement and failure rates by model or track

Exit condition:

- multiple models can be compared on the same dataset slice with reusable benchmark summaries

### Priority 4: Research Dataset Quality

Goal:

- improve coverage quality before tightening benchmark policy

Tasks:

1. keep the `5-turn` generation workflow as the current research default and treat `7-turn` as an explicit follow-up experiment, not the baseline
2. continue comparing `gpt-oss:20b`, `gemma3:27b`, and `gemma3:4b` on the same `5-turn` scenario pack before scaling to larger research corpora
3. review `v2` preview samples and remove obviously synthetic or low-signal patterns
4. widen scenario diversity without destabilizing the stable `v1` regression pack

Exit condition:

- richer datasets are useful for research loops and manual review, while `v1` remains a clean regression floor

Current checkpoint:

- `5-turn` research generation produced usable small-sample outputs on `gpt-oss:20b` and `gemma3:27b`
- `5-turn` research generation now also completed `300-item` runs for `gpt-oss:20b`, `gemma3:27b`, and `gemma3:4b`
- `7-turn` generation remains exploratory because the current prompts introduce too much repetition
- the completed `300-item` generation experiments are:
  - `remote-5turn-gptoss20b-300`
  - `local-5turn-gemma27b-300`
  - `remote-5turn-gemma4b-300`
- the next restart should use the exact commands captured in `docs/OPERATIONS.md`

### Priority 5: Few-Shot To Low-Cost ML

Goal:

- test whether a cheaper learned baseline can replace some provider-backed research loops

Tasks:

1. define exportable training rows from stored few-shot or benchmark artifacts
2. lock down train/validation/test usage for lightweight learned observers
3. add one low-cost ML baseline track that can run through the same JDVP extraction contract
4. benchmark the learned baseline against heuristic, few-shot, and provider-backed tracks

Exit condition:

- one cheap learned observer can be trained and evaluated through the same benchmark path as the existing tracks

### Priority 6: Experiment Management Boundary

Goal:

- decide how far POCv3 should go toward POCv2-style experiment management

Tasks:

1. keep SQLite limited to cataloging unless recovery needs justify broader orchestration
2. evaluate whether dataset generation and JDVP result sets should become first-class experiment entities
3. postpone heavier process orchestration until lightweight catalog operations stop being enough

Exit condition:

- the project has a clear boundary between lightweight cataloging and more expensive orchestration work

## Working Sequence

Use this order for the next implementation passes:

1. run a 100-item local `llm_turn_simulated` generation trial and inspect reject/failure breakdowns
2. dataset generation and preview review
3. model-by-model test runs on shared dataset slices
4. ensemble summaries and model benchmarking
5. few-shot-derived low-cost ML baseline exploration

Current next task:

- run a local 100-item `llm_turn_simulated` generation experiment and tune model/prompt/quality-gate settings from the resulting reject/failure profile

## Technical Debt And Quality Backlog

Discovered during full-project review on 2026-03-20.

### High Priority

1. **dataset_run_service: broaden per-item exception handling**
   - `dataset_run_service.py` catches only `ServiceError` in the per-item loop; any other exception aborts the entire batch
   - fix: catch `Exception` per item, record in failure rows, let the batch complete

2. **eval: decouple CORE_FIELDS from hardcoded tuple**
   - `ensemble_benchmark.py` hardcodes `CORE_FIELDS` instead of deriving from protocol schema or `enums.py`
   - fix: import from `protocol_core.enums.CORE_FIELD_NAMES` or validate against loaded schema

3. **docs: document undocumented experiments and scripts**
   - OPERATIONS.md is missing: gemma12b-300, gemma12b-smoke1 dataset runs
   - OPERATIONS.md is missing scripts: `list_generation_runs.py`, `list_failed_generation_items.py`, `rerun_failed_generation_items.py`, `validate_contracts.py`, `prepare_remote_turn_sim_trial100_matrix.py`

### Medium Priority

4. **service: add catalog upsert error handling**
   - `poc_service.py` and `dataset_run_service.py` do not catch SQLite errors on `catalog.upsert_*()` calls
   - fix: wrap in try/except with logging so runs are not silently untracked

5. **eval: add file I/O error handling**
   - `ensemble_benchmark.py` `_load_run_manifest()` and `_load_extracts()` have no try/except
   - fix: catch `FileNotFoundError` / `JSONDecodeError` with descriptive context

6. **README status date stale**
   - last updated 2026-03-16; should reflect latest project state

7. **test coverage gaps for protocol_core**
   - no direct tests for `build_jsv()`, `build_jsv_from_hint()`, or `CanonicalSchemaValidator`
   - currently covered indirectly through integration tests; direct unit tests would improve confidence

### Low Priority

8. **poc_service: narrow _git_revision() exception scope**
   - broad `except Exception` silently returns `"workspace-local"` on any failure
   - fix: catch `subprocess.CalledProcessError` and `FileNotFoundError`, log others

9. **run_validation_suite: import check_assistant_docs_sync instead of subprocess**
   - currently spawns a subprocess; could import directly like `validate_contracts_main()`

10. **dataset_run_service: avoid absolute-path dataset_run_id**
    - `_build_dataset_run_id()` uses `Path.resolve()` as ID; breaks if paths change
    - consider UUID or content-hash alternative

11. **dataset generate_dataset: narrow broad exception catch**
    - line 732 catches all `Exception` during interaction generation; masks programming errors
    - fix: catch specific exceptions (`ValueError`, `RuntimeError`)

12. **eval: add logging infrastructure**
    - eval modules use `print()` only; no structured logging for long-running benchmarks

## Deferred Automation

Only start this phase after M1-M8 justify the extra complexity.

Deferred tasks:

- silver generation
- bootstrap gold
- release gating
- artifact registries
- scheduled jobs
