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

Exit condition:

- one LLM track runs through the same pipeline without special-casing protocol logic

## Phase 4: M4 Comparative Evaluation

Goal:

- compare methods without changing canonical artifact semantics

Tasks:

1. add overlay disagreement report
2. add fixture batch comparison runner
3. add simple per-field agreement metrics
4. add protocol-alignment regression report

Exit condition:

- track comparison is available as overlay output

## Phase 5: Deferred Research Automation

Only start this phase after M1-M4 are stable.

Deferred tasks:

- silver generation
- bootstrap gold
- release gating
- artifact registries
- scheduled jobs
