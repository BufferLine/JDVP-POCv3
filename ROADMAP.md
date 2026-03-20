# ROADMAP

Quick-reference kanban. Full detail: `docs/IMPLEMENTATION_PLAN.md`

Updated: 2026-03-20

## Done

- [x] M1 Protocol-core generation + canonical schema validation
- [x] M2 Deterministic heuristic baseline extraction
- [x] M3 Provider-backed LLM observer + few-shot prompt track
- [x] M4 Run storage, per-turn extracts, manifests, resumable checkpoints
- [x] M5 Disagreement reporting, ensemble comparison, few-shot regression
- [x] M6 Dataset generation, v2 scenario packs, preview generation
- [x] M7 Few-shot pack creation + benchmark planning/execution
- [x] M8 Service boundary, JSON transport, validation unification
- [x] SQLite catalog for datasets and run state
- [x] Failed-run inspection and rerun scripts
- [x] Multi-turn (5-turn) research generation: gptoss20b/gemma27b/gemma4b 300-item
- [x] Local 100-item llm_turn_simulated trial
- [x] Technical debt cleanup (shared_utils consolidation)

## In Progress

- [ ] 7-turn generation tuning (repetition issue in current prompts)

## Backlog — Features

- [ ] P1: Operational recovery — catalog-driven resume, scenario/track failure slicing
- [ ] P2: Dataset-scoped execution — batch runner, dataset-level metadata
- [ ] P3: Model benchmarking — model-level summaries, leaderboard view
- [ ] P4: Research dataset quality — scenario diversity, v2 preview cleanup
- [ ] P5: Few-shot to low-cost ML — training rows, learned observer benchmark
- [ ] P6: Experiment management boundary — catalog scope decision

## Backlog — Tech Debt (High)

- [ ] `dataset_run_service`: catch all exceptions per item, not just ServiceError
- [ ] `ensemble_benchmark`: derive CORE_FIELDS from `protocol_core.enums` instead of hardcoding
- [ ] OPERATIONS.md: document gemma12b runs + 5 undocumented scripts

## Backlog — Tech Debt (Medium)

- [ ] `poc_service` / `dataset_run_service`: catalog upsert error handling
- [ ] `ensemble_benchmark`: file I/O error handling for `_load_run_manifest` / `_load_extracts`
- [ ] README.md: update status date (stale since 2026-03-16)
- [ ] protocol_core: direct unit tests for `build_jsv`, `build_jsv_from_hint`, `CanonicalSchemaValidator`

## Backlog — Tech Debt (Low)

- [ ] `poc_service._git_revision()`: narrow exception scope
- [ ] `run_validation_suite`: import `check_assistant_docs_sync` instead of subprocess
- [ ] `dataset_run_service._build_dataset_run_id()`: replace absolute-path ID with UUID/hash
- [ ] `generate_dataset`: narrow broad `except Exception` at line 732
- [ ] eval modules: add structured logging (replace `print()`)

## Deferred

- Silver generation
- Bootstrap gold
- Release gating
- Artifact registries
- Scheduled jobs
