# MVP Measurement Architecture

## Product Claim

The project covers Agent(Human|AI)–Agent(Human|AI); the implemented MVP substrate is Human–AI v1.5. Develop a low-cost detector and a critical-thinking measurement framework through the [automated research loop](./AUTOMATED_RESEARCH_LOOP.md), without a mandatory human-expert stage.

The initial loop should demonstrate a narrow, testable capability:

> Semantic interaction data can be used to extract observable patterns of judgment delegation, delegation awareness, cognitive engagement, and information seeking at low cost.

It must not claim to diagnose a person's intelligence, personality, or general critical-thinking ability. A critical-thinking practice profile is a later, separately validated interpretation of repeated measurements.

## Layer Boundaries

| Layer | Question answered | Output | Status |
| --- | --- | --- | --- |
| Canonical JDVP | What was observed on each turn? | JSV, DV, Trajectory | JDVP v1.5 canonical artifacts |
| Observer measurement | How was it observed, and how reliable is the observation? | track output, evidence, model/calibrator provenance, uncertainty | POCv3 overlay |
| Interaction summary | What changed across the conversation? | trend, volatility, decision-relevant basis | JDVP v1.6 proposal |
| Product interpretation | What feedback is useful to the participant? | dependency/reflection signals, practice profile | MVP tool layer |
| Assessment and licensing | What may be standardized or certified? | validated rubric, calibration pack, conformance procedure | post-validation only |

Canonical JDVP fields remain descriptive. Calibration artifacts, continuous scores, and critical-thinking interpretations must stay in the observer or product layer until the protocol formally adopts a compatible extension.

## Cost-Quality Architecture

```text
semantic interaction data
  -> explicit trace / rules
  -> embedding + calibrated four-axis classifier (<100 ms target)
  -> uncertainty or decision-relevance gate
  -> selective high-capability reference AI and challenge pass
  -> auditable JDVP artifacts and participant-facing feedback
```

The embedding classifier is a first-stage detector, not a replacement for validation. It should record a model artifact ID, label provenance, evidence, latency, and uncertainty on every result. A high-capability observer labels samples and reviews uncertain cases. Preserve AI-reference provenance and separate detector fidelity from framework validity; sample authorized external behavior/outcomes when text is insufficient.

## Current Feasibility Evidence

The current embedding calibration experiment uses the previously fine-tuned `jdvp-embedding-v1` encoder and a four-axis linear calibration head.

- In-sample capacity diagnostic: 401 non-contested synthetic turns, four axes, no Ridge regularization.
- Per-axis rounded exact agreement: JH 94.0%, DA 90.3%, CE 72.1%, IS 75.3%.
- All-four-axis exact agreement: 56.1%.
- High-level (`>=7`) training-set detection had no false positives or false negatives.
- Sample warm-path extraction was below 100 ms per turn; this is a feasibility measurement, not a production latency SLO.
- Interaction-held-out synthetic evaluation remains materially weaker (average MAE about 2.08); it is the relevant current quality signal.

These results establish representation capacity and low-latency feasibility. They do not establish out-of-sample accuracy, human agreement, or critical-thinking construct validity.

## Data And Label Strategy

Scale data by information value, not only volume.

1. Build diverse semantic interactions, especially decision-relevant turns and counterexamples that distinguish task execution from judgment delegation.
2. Store independent labels from strong observers with model/prompt/version provenance and disagreement, rather than flattening them into an unexplained gold label.
3. Route high-disagreement, high-impact, and boundary cases to independent reference labeling and an AI challenge pass. Preserve dissent; do not replace it with an unmarked majority vote.
4. Keep train/dev/test splits at the interaction level. Prevent the same conversation or template family from crossing the split boundary.
5. Train the low-cost classifier on the declared label tier, then report per-axis MAE, ordinal/bucket accuracy, false-positive and false-negative rates, calibration, latency, and cost on an untouched holdout set.

Select and version the reference model explicitly. Qualify it with counterexamples, invariance tests, reserved domains, and sampled external criteria. Human-expert approval is not required. Same-model agreement does not establish independent validity.

## MVP Readiness Gates

Before releasing a participant-facing MVP, the project should have:

- a versioned Human-AI measurement profile and a decision-relevance policy;
- a documented low-cost observer artifact with reproducible inference;
- an interaction-level untouched evaluation set;
- a versioned AI-reference anchor set with challenge results and unresolved cases, plus sampled external criteria;
- observer disagreement, false-positive/false-negative, latency, and unit-cost reports;
- uncertainty-based escalation to a higher-capability reference observer and challenge pass;
- participant-facing language that describes interaction patterns, not personal traits or diagnoses.

The JDVP v1.6 proposal should formalize the profile, observer provenance, uncertainty, decision-relevant turn marker, and interaction summary only after the extension is reviewed in the protocol repository.

## Planned Profile Expansion

Human–Human adds authorized relationship records for evidence-backed scam, coercive/cult recruitment, and gaslighting pattern review. AI–AI adds typed agent/tool/action traces for orchestration experiments. Common candidate observables include authority movement, independent evidence, verification, and revision; interpretation and thresholds remain profile-specific. These are planned contracts, not capabilities of the existing human_input/ai_response schema. See R4–R6 in the implementation plan.
