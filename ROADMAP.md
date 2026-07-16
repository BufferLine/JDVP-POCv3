# ROADMAP

Quick-reference kanban. Full detail: `docs/IMPLEMENTATION_PLAN.md`

Updated: 2026-07-15

## Done

### Infrastructure & Pipeline
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

### Research — Level Extraction (2026-03-31 ~ 04-03)
- [x] 0-10 level scoring validation — extreme r=0.97, real data r=0.56 (turn-level limit confirmed)
- [x] JDVP prompt bias discovery — v1-v3 prompts cause JH=Shared bias, free-form labeling works better
- [x] Prompt optimization — v4b (contrast examples + v2 field names, ~390 tokens) optimal
- [x] DA field difficulty — self-consistency r=0.47, inherently low observability
- [x] gemma4:e4b local viability — r=0.96 vs sonnet-4, free local alternative

### Research — 3-Axis DA Derivation (2026-04-06 ~ 07)
- [x] DA derivation formula — DA=0.162·JH+0.570·CP+0.268·ID, derived > direct measurement
- [x] CoT prompt optimization — `level_observer_3axis_cot.txt`, small model CP std 0.6→1.4
- [x] 6-model 300-conversation comparison — gpt4.1-cot baseline, gemma4:26b local best (DV상관 0.53)
- [x] Conversation trend analysis — 4-model unanimity: rising 37, stable 65, falling 15
- [x] ShareGPT 300 decision conversation curation

### Research — Embedding Distillation (2026-04-06 ~ 08)
- [x] MiniLM-L6-v2 contrastive fine-tune — DA R²=0.446, High DA F1=0.39, 5ms inference
- [x] 8-model embedding benchmark — pre-trained: nomic(0.207) > bge-base(0.191) > gte-base(0.156) > MiniLM(0.082)
- [x] Fine-tune dominance confirmed — fine-tuned MiniLM(0.446) >> best pre-trained(0.207)

### Research — Protocol Review & v1.6 Proposal (2026-07-15)
- [x] Protocol completeness review — psychology/psychometrics/engineering/governance perspectives (`docs/research/2026-07-15-protocol-completeness-review.md`)
- [x] v1.6 proposal draft (JDVP-2026-002) — Interaction Summary artifact, derived DA, decision-relevant marker, observer profile (`docs/research/jdvp-protocol-v1.6-conversation-unit-proposal.md`)
- [x] v1.5 migration gap identified — upstream adopted level scoring, POCv3 core still v1.4 (issue #15)
- [x] Positioning draft "From Tests to Traces" — test→trace psychometric argument, LLM-annotator literature, metrology thesis (`docs/research/2026-07-15-from-tests-to-traces-positioning.md`)
- [x] Long-arc vision note — dyadic generalization (agent↔agent), design guardrails, sequencing (`docs/research/2026-07-15-long-arc-vision-dyadic-generalization.md`)

## In Progress

- [ ] Embedding fine-tune on bge-base/bge-small/MiniLM — contrastive learning, 15K pairs, 4-model consensus
- [ ] Embedding trend accuracy check — conversation-level rising/stable/falling vs LLM ground truth
- [ ] 7-turn generation tuning (repetition issue in current prompts; smoke10 pack exists)

## Backlog — Embedding & Cheap Inference

- [ ] Reasoning-augmented fine-tune — use LLM reasoning field as training signal
- [ ] Embedding trend classifier — turn-level DA predictions → conversation-level trend (rising/stable/falling)
- [ ] 3-tier architecture design — embedding screening(5ms) → gemma4:e4b(2s) → gpt4.1(API)

## Backlog — Silver Labeling & Data Scale

- [ ] Silver baseline labeling — gpt4.1-cot full 300-conversation reference labels
- [ ] gemma4:26b large-scale comparison — local labeling quality validation
- [ ] Data scale-up — expand beyond 300 conversations for embedding training

## Backlog — Product Direction

- [ ] Trend-based JDVP tracking service design — direction over precision
- [ ] Human evaluation protocol — trend accuracy validation with real users

## Backlog — Features (Legacy)

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

- Bootstrap gold
- Release gating
- Artifact registries
- Scheduled jobs
