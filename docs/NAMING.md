# POCv3 Naming Guide

This document defines the working names used in POCv3 so dataset, run, benchmark, and track terms do not drift.

## Tracks

### `fixture_hint`

Meaning:

- uses `meta.jsv_hint` already embedded in the input turn
- not a learned or inferred model
- canonical reference path for fixtures and smoke runs

Use when:

- validating protocol-core generation
- building stable fixture-derived packs

### `heuristic_baseline`

Meaning:

- deterministic rule-based extractor
- lowest-cost inferred baseline

Use when:

- creating a stable non-LLM regression floor
- comparing other methods against a predictable baseline

### `llm_observer`

Meaning:

- provider-backed observer using the shared LLM normalization boundary
- raw prompt-based JDVP extraction without few-shot prompt injection

Use when:

- testing direct model behavior
- comparing provider-backed extraction against heuristics or richer prompting

### `fewshot_prompt`

Meaning:

- still an LLM-backed track
- uses a few-shot pack as prompt context
- this is in-context learning, not model training

Use when:

- testing whether prompt examples improve extraction quality
- running few-shot benchmarks against a shared baseline

### `cheap_ml_baseline`

Meaning:

- lightweight learned baseline trained from few-shot examples
- currently a Naive Bayes classifier over the pack examples
- this is a cheap model path, not an LLM prompt path

Use when:

- testing whether a low-cost local model can replace some LLM research loops
- creating a learned baseline that shares the same JDVP extraction contract

## Dataset Packs

### `v1` dataset packs

Meaning:

- stable regression-oriented dataset packs
- intended to remain predictable enough for CI and repeated validation

Current examples:

- `config/datasets/general_scenarios_v1.json`
- `generated/synthetic-general/v1`

### `v2` dataset packs

Meaning:

- richer research-oriented dataset packs
- may include more varied wording, mixed blueprints, and less stable patterns
- should be reviewed before influencing the stable baseline

Current examples:

- `config/datasets/general_scenarios_v2.json`
- preview datasets built from the `v2` pack

## Run Types

### single run

Meaning:

- one interaction through one track
- output is one run directory with canonical artifacts and extracts

Typical command:

- `src.pipeline.run_poc`

### dataset run

Meaning:

- one dataset slice through one track
- output is one `dataset_run_summary.json` plus one run directory per interaction

Typical command:

- `scripts/run_dataset.py`

### dataset matrix run

Meaning:

- one dataset slice through multiple track or model configurations
- output is one `dataset_matrix_summary.json`

Typical command:

- `scripts/run_dataset_matrix.py`

### benchmark matrix

Meaning:

- pairwise or ensemble comparison built from a dataset matrix run
- usually compares one baseline label against one or more comparison labels

Typical command:

- `scripts/benchmark_dataset_matrix.py`

## File Names

### `preview.json`

Meaning:

- human-readable preview of generated dataset items
- for manual quality review, not canonical benchmark output

### `dataset_run_summary.json`

Meaning:

- summary for one track over one dataset slice

### `dataset_matrix_summary.json`

Meaning:

- summary for multiple track/model configurations over the same dataset slice

### `benchmark_matrix_summary.json`

Meaning:

- comparison summary derived from a dataset matrix run

### `benchmark_results.json`

Meaning:

- few-shot benchmark result set
- usually produced by `src.eval.run_fewshot_benchmark`

## Recommended Run ID Patterns

Use short IDs that explain scope:

- single run: `<track>-<interaction_id>`
- local smoke run: `smoke-local`
- local baseline run: `baseline-local`
- retry run: `<original-run-id>-retry-01`

Avoid:

- generic ids like `test1`, `run2`, `tmp`

## Naming Distinctions To Preserve

Keep these distinctions explicit:

- `fewshot_prompt` is an LLM path
- `cheap_ml_baseline` is a learned cheap-model path
- `v1` is stable baseline-oriented data
- `v2` is richer research-oriented data
- dataset runs produce JDVP outputs
- benchmark runs compare JDVP outputs
