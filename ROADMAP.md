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

## Backlog — Bugs (Codex review, 2026-03-20, all reproduced)

### Critical

- [x] `dv_ordinal.py:67`: non-general context DV emits empty `extensions: {}` → removed (delta extensions not yet implemented)
- [x] `schema_validate.py:44`: no `format_checker` → FormatChecker added to all validators
- [x] `trajectory.py:18` vs `raw_interaction.schema.json:45`: raw schema minItems changed to 2
- [x] `poc_service.py:152` + `run_storage.py:49`: context_turns now uses processed_turns list instead of index slicing
- [x] `cheap_ml_baseline.py:107`: empty pack raises ValueError + artifact load validates field completeness
- [x] `ensemble_benchmark.py:92`: keyed by `run_id:track_name` instead of `track_name` alone
- [x] `ensemble_benchmark.py:106`: skipped turns now tracked in summary (turns_skipped, skipped_turn_details)
- [x] `run_dataset.py:30`: exits with code 1 when ok=false
- [x] `generate_dataset.py:605`: per-item deterministic RNG seeding replaces shared RNG
- [x] `run_fewshot_regression_suite.py:57`: path guard prevents rmtree on project root or ancestors

### Medium

- [x] `poc_service.py:357`: distinguishes input_not_found vs file_not_found based on pipeline stage
- [x] `json_api.py:23`: _parse_bool() correctly handles string "false"/"true"
- [x] `errors.py:12`: __post_init__ calls super().__init__(self.message)
- [x] `dataset_run_service.py:71`: dataset_run_id now includes track_name and split
- [x] `fewshot_prompt.py:34`: logs warning and sets _zero_shot_fallback flag when 0 examples
- [x] `llm_observer.py:89`: non-JSON body wrapped in RuntimeError with context
- [x] `eval_service.py:80`: JSONDecodeError caught before ValueError → benchmark_input_invalid
- [x] `generate_dataset.py:782`: pending_count now subtracts rejected_count
- [x] `rerun_failed_runs.py:21`: retry ID now checks for existing retries and increments
- [ ] JSON API + validation suite: malformed payloads and pytest/upstream paths untested

## Backlog — Tech Debt (High)

- [x] `dataset_run_service`: catch all exceptions per item (already catches Exception)
- [x] `ensemble_benchmark`: derive CORE_FIELDS from `protocol_core.enums` instead of hardcoding
- [ ] OPERATIONS.md: document gemma12b runs + 5 undocumented scripts

## Backlog — Tech Debt (Medium)

- [x] `poc_service` / `dataset_run_service`: catalog upsert error handling (_safe_catalog_upsert)
- [x] `ensemble_benchmark`: file I/O error handling for `_load_run_manifest` / `_load_extracts`
- [x] README.md: update status date to 2026-03-25
- [x] protocol_core: 5 direct unit tests for `build_jsv`, `build_jsv_from_hint`, `CanonicalSchemaValidator`

## Backlog — Tech Debt (Low)

- [x] `poc_service._git_revision()`: narrowed to CalledProcessError/FileNotFoundError/OSError
- [x] `run_validation_suite`: import `check_assistant_docs_sync` directly
- [x] `dataset_run_service._build_dataset_run_id()`: replaced with hash-based `dsr-{label}-{hash}` ID
- [x] `generate_dataset`: narrowed to ValueError/RuntimeError/OSError/JSONDecodeError
- [x] eval modules: print() only in CLI main(), no library-level issue

## Deferred

- Silver generation
- Bootstrap gold
- Release gating
- Artifact registries
- Scheduled jobs
