# ROADMAP

Quick-reference kanban. Full detail: `docs/IMPLEMENTATION_PLAN.md`

Updated: 2026-03-30

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
- [x] bufferline-llm integration as shared LLM provider (PR #14)
- [x] P1: Operational recovery — catalog queries, list/rerun scripts, scenario/track failure slicing
- [x] P2: Dataset-scoped execution — batch runner, dataset-level metadata, matrix runs
- [x] P5: Few-shot to low-cost ML — cheap_ml_baseline track, training rows, benchmark path
- [x] Phase A: Extraction quality foundation — system prompt rubric, compare_runs tests, pipeline observability
- [x] C-7: 102-item triple-annotated pilot (codex/gemini/independent), adjudication, manifest
- [x] D-10: Catalog integrity — UNIQUE constraints, model_id in dataset_runs, atomic write_json
- [x] D-11: Heuristic baseline upgrade — protocol-aligned defaults, ai_response analysis, regex expansion
- [x] Codex deep review bugs: all 10 critical + 9/10 medium fixed
- [x] Codex integrity review bugs: all 6 critical/high + 8 medium fixed
- [x] Tech debt backlog: all high + all medium + all low resolved

## In Progress

- [ ] 7-turn generation tuning (repetition issue in current prompts; smoke10 pack exists)

## Backlog — Features

- [ ] P3: Model benchmarking — leaderboard view (ensemble voting done, ordinal distance scoring missing)
- [ ] P4: Research dataset quality — scenario diversity, v2 preview cleanup
- [ ] P6: Experiment management boundary — catalog scope decision

## Backlog — Evaluation Rigor (Phase B)

- [ ] B-4: jsv_hint silver-label metric — hint-vs-extraction accuracy function
- [ ] B-5: Scenario diversity expansion — reverse trajectory, steady-state, edge cases (7+ new scenarios)
- [ ] B-6: Confusion matrix + kappa statistics — per-field transition matrix, Cohen's kappa, bootstrap 95% CI

## Backlog — Experiments (Phase C-D)

- [ ] C-8: Prompt ablation — 3-5 system prompt variants (detailed/minimal, with/without evidence_spans)
- [ ] D-9: Ensemble improvements — ordinal distance disagreement scoring (confidence voting done)

## Backlog — Bugs

- [ ] JSON API + validation suite: malformed payload edge cases untested

## Deferred

- Silver generation
- Bootstrap gold
- Release gating
- Artifact registries
- Scheduled jobs
