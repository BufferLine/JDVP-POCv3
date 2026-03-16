# AGENTS.md

Sync note: Keep all content below this line identical between `AGENTS.md` and `CLAUDE.md` (title line may differ). Update both in the same commit.

## Project
JDVP POCv3 is a core-first implementation workspace for canonical JDVP artifact generation, extraction-track evaluation, dataset-driven research loops, and lightweight operational recovery.

## Repository Structure
- Core protocol logic: `src/protocol_core/`
- Extraction tracks: `src/method/tracks/`
- Pipeline and run storage: `src/pipeline/`
- Service-facing orchestration: `src/service/`
- Dataset generation and few-shot tooling: `src/dataset/`
- Evaluation and benchmarking: `src/eval/`
- Operations and roadmap docs: `docs/`
- Root orchestration scripts: `scripts/`

## Scope
- Canonical JSV, DV, and Trajectory generation aligned to JDVP `v1.4`
- Deterministic, provider-backed, few-shot, and cheap-ML extraction tracks
- Dataset generation for stable regression packs and richer research packs
- Dataset-scoped execution, matrix runs, and benchmark summaries
- Lightweight SQLite cataloging for datasets and JDVP run recovery

## Source Of Truth
- Protocol semantics originate in the sibling protocol repository:
  - `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/protocol.md`
  - `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/docs/whitepaper.md`
  - `/Users/sangwon0001/Projects/bufferline/JDVP-protocol/v1/`
- Vendored local schema snapshot:
  - `vendor/JDVP-protocol/v1/schemas/`
  - `vendor/JDVP-protocol/v1/schema_snapshot.json`

## Non-Goals
- No heavyweight workflow engine yet
- No web dashboard or broad experiment UI
- No database-first artifact storage migration
- No production HTTP transport until a real consumer requires it

## Editing Rules
- Preserve canonical JDVP semantics and schema alignment.
- Keep protocol-core logic independent from extraction strategy.
- Keep stable regression data and richer research data explicitly separated.
- When adding or renaming tracks, update:
  - `src/method/tracks/factory.py`
  - CLI choices in `src/pipeline/run_poc.py`
  - `docs/NAMING.md`
  - README usage examples when relevant
- When adding dataset or benchmark flows, update the roadmap and operations docs in the same change when the workflow meaning changes.
- Keep `AGENTS.md` and `CLAUDE.md` synchronized.

## Naming And Documentation
- Naming source of truth: `docs/NAMING.md`
- Roadmap source of truth: `docs/IMPLEMENTATION_PLAN.md`
- Operational usage and validation flow: `docs/OPERATIONS.md`
- README should stay concise and point to the more detailed docs instead of duplicating them.

## Current Working Distinctions
- `fewshot_prompt` means LLM in-context learning with prompt examples.
- `cheap_ml_baseline` means a lightweight learned model path, not an LLM path.
- `v1` datasets are stable regression-oriented packs.
- `v2` datasets are richer research-oriented packs.
- `dataset run` means one track over one dataset slice.
- `dataset matrix` means multiple track/model configurations over the same dataset slice.
- `benchmark matrix` means comparison outputs derived from a dataset matrix.

## Operational Rules
- Default validation entrypoint: `python3 scripts/run_validation_suite.py`
- Default catalog path: `data/catalog/pocv3.sqlite3`
- Recovery tools:
  - `python3 scripts/list_failed_runs.py`
  - `python3 scripts/rerun_failed_runs.py`
- Dataset execution tools:
  - `python3 scripts/run_dataset.py`
  - `python3 scripts/run_dataset_matrix.py`
  - `python3 scripts/benchmark_dataset_matrix.py`
- Cheap ML training setup:
  - `python3 scripts/train_cheap_ml.py`

## Recommended Validation
- `python3 -m pytest -q`
- `python3 scripts/run_validation_suite.py --allow-missing-upstream`
- `python3 scripts/check_protocol_schema_sync.py --require-upstream`
