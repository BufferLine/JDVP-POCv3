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

## Deferred Automation

Only start this phase after M1-M8 justify the extra complexity.

Deferred tasks:

- silver generation
- bootstrap gold
- release gating
- artifact registries
- scheduled jobs
