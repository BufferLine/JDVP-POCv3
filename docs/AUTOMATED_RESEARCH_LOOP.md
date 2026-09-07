# Automated Measurement Research Loop

Status: planned, 2026-09-07. This document specifies the next implementation; it does not describe an existing autonomous runner. Product scope is Agent(Human|AI)–Agent(Human|AI). Human experts are excluded as a required process stage.

## Existing Building Blocks

PR #21 supplies canonical v1.5 artifacts, embedding_screen, embedding_calibrated, persisted training heads, grouped-by-interaction holdout, calibration reporting, and latency/estimated-cost metadata. PR #19 supplies pairwise sign-flip/boundary breakdowns. Neither implements the full loop or new relationship input schemas.

Known gaps: calibrated observer confidence currently derives from artifact-wide MAE, not input uncertainty; embedding tracks do not use prior conversation content; evidence currently includes an input excerpt rather than an axis-specific explanation. The training split does not enforce template-family/domain isolation. The calibration report does not yet integrate all chance-corrected and bootstrap research metrics. Cost metadata can be a configured estimate, not measured billing or total operating cost.

## Iteration Contract

Each iteration must record framework/profile versions, participant types, dataset and split hashes, sampling policy/probabilities, reference model and prompt versions, detector artifact identity, conditioning policy, seed, budget, and parent iteration. Freeze the input manifest before reference labeling. Keep canonical v1.5 output separate from proposed profile metadata.

1. **Sample:** combine a random audit slice with uncertainty, reference disagreement, boundary, and novel-domain slices. Declare quotas and selection probabilities; do not report targeted-sample error as population error without adjustment.
2. **Label independently:** a high-capability reference (Astra is a candidate, selected explicitly at execution time) labels without seeing detector predictions or prior adjudication. Record axis-specific evidence, uncertainty, and unavailable context.
3. **Challenge:** a separate pass/model probes alternative interpretations and rubric violations. Store both judgments and a resolution or abstention. Same-model passes are not independent replications. Teacher claims are not human gold.
4. **Test the framework:** generate style-preserving paraphrases and behavior-changing pairs; distinguish deliberate delegation from passive acceptance, substantive verification from performative disagreement, and independent consensus from copied evidence. Use authorized external action/outcome samples where the text cannot resolve the question. Missing evidence remains missing.
5. **Train:** distill declared reference labels into the low-cost observer. Isolate conversations, template families, and reserved domains across splits. Record encoder training provenance too; a head-only holdout cannot rule out encoder exposure.
6. **Evaluate:** run a frozen evaluation set and baselines: deterministic rules, uncalibrated retrieval, direct reference analysis, presence/absence indicators, and single-turn versus trajectory ablation. Keep detector fidelity and framework validity reports separate.
7. **Promote or retain:** apply predeclared per-profile thresholds and regression limits. Automatic promotion means selecting a research artifact, not deploying a product or publishing a standard. Retain the previous artifact on failure. A changed rubric starts a new framework version; retire exposed test sets into development and acquire a fresh reserve.

## Metrics and Budget

Report per-axis MAE, ordinal agreement, precision/recall and false positives/negatives at declared thresholds, chance-corrected agreement, and conversation-clustered confidence intervals. Include abstention, coverage, escalation rate, cold/warm and end-to-end latency, and total cost including reference/challenge calls, retries, training, and external-sample checks. Do not present cosine similarity or 1-minus-average-MAE as a calibrated probability.

Report framework counterexample failures, paraphrase invariance, response to controlled changes, held-out domain performance, and association with sampled observable criteria. Interventions with controls are needed for claims of causal improvement. An internally defined score increasing is not an external success criterion.

Before execution, set iteration/token/spend ceilings, retry limits, stop conditions, and permitted data/model sources. Stop on budget exhaustion, missing required evaluation evidence, repeated regressions, or material rubric drift. Recovery should reuse completed version-matched artifacts through the current file/catalog boundary; no heavyweight workflow engine is required.

## Relationship Experiments

- **Human–AI:** first end-to-end loop using the existing schema, with actual authorized conversations separated from synthetic challenge packs.
- **AI–AI:** propose typed participants and evidence/tool/action references upstream; collect controlled orchestration traces. Compare task success, error propagation, recovery, calls, and cost with/without measurement-driven interventions. Vary agent models and task families.
- **Human–Human:** propose a separate profile and authorized corpus. Test benign counterexamples, contextual omissions, and evidence for isolation/authority pressure/judgment invalidation. Output tentative pattern evidence rather than definitive scam, cult, or gaslighting classifications.

A shared axis survives only if its interpretation and utility are supported in each tested profile. Profile-specific axes and non-comparable outputs are valid outcomes.
