# JDVP Protocol Enhancement Proposal: Conversation-Unit Measurement

**Proposal ID**: JDVP-2026-002
**Date**: 2026-07-15
**Author**: sangwon0001
**Status**: Draft — for review before PR to JDVP-protocol
**Evidence**:
- [`2026-04-06-derived-da-study.md`](./2026-04-06-derived-da-study.md) — 3-axis extraction, DA derivation, trend validation (6 models × 300 conversations × 4,103 turns)
- [`2026-03-31-level-extraction-study.md`](./2026-03-31-level-extraction-study.md) — turn-level self-consistency ceiling
- Multi-perspective protocol completeness review ([JDVP-POCv3 `docs/research/2026-07-15-protocol-completeness-review.md`](https://github.com/BufferLine/JDVP-POCv3/blob/main/docs/research/2026-07-15-protocol-completeness-review.md))
**Companion**: [`MVP_MEASUREMENT_ARCHITECTURE.md`](../MVP_MEASUREMENT_ARCHITECTURE.md) — implementation boundary and testing path for low-cost observers
**Revision note (2026-07-16)**: reliability figures corrected against the V3 Final Benchmark Report ([JDVP-POCv3 `docs/research/2026-07-16-v3-final-benchmark-report.md`](https://github.com/BufferLine/JDVP-POCv3/blob/main/docs/research/2026-07-16-v3-final-benchmark-report.md)), which applied this proposal's own Change 5 reporting standards to the evidence data. Two claims were corrected: raw trend agreement is now quoted chance-corrected (weighted κ≈0.51 over a 51.7% majority baseline), and "near-zero direct contradictions" is retracted (sign reversals are 13.7% of disagreements among strong observers).

---

## Summary

This proposal promotes the conversation (interaction) to a first-class measurement unit. It adds a canonical **Interaction Summary** artifact (trend, mean level, volatility, peak per dimension), reclassifies `delegation_awareness` as a **derived observable**, introduces a **decision-relevant turn** marker, upgrades confidence reporting from nominal labels to optional numeric uncertainty with an observer profile block, and defines **measurement reporting standards** (chance-corrected agreement, resampled confidence intervals, observer-conditioning disclosure, ordinal-robust aggregation).

The MVP may use an embedding-calibrated classifier as one observer implementation, but its continuous scores, model artifact, retrieval state, and escalation policy remain observer-layer metadata. This proposal does not add a critical-thinking score to canonical JDVP; that interpretation remains a separately validated product layer.

v1.5 documented the turn-level self-consistency ceiling (r≈0.56 on natural data) as a known limit but standardized no response to it. Our subsequent experiments show that conversation-level direction is the unit implementations actually rely on, and that it degrades more gracefully than position — disagreements concentrate at the stable boundary rather than reversing sign. Its reliability is nonetheless moderate once chance-corrected (weighted κ≈0.51), which is precisely why the artifact must carry the reporting standards of Change 5 rather than travel with raw percentages. v1.6 makes that operative unit canonical, with honest uncertainty attached.

## Motivation

### Problem 1: The protocol's most reliable signal has no canonical artifact

Empirical results from the 6-model × 300-conversation study:

| Measurement unit | Inter-model reliability |
|---|---|
| Turn-level absolute levels | r=0.41–0.61 vs baseline; ±2 typical disagreement |
| Turn-level DV correlation | r=0.36–0.53 |
| Conversation-level trend (rising/stable/falling) | weighted κ≈0.51 [0.42–0.61] vs reference (raw 67.7% over a 51.7% majority-class baseline); 6-observer ordinal Krippendorff α=0.30; ~82–86% of disagreements sit on the rising/stable boundary (slope ≈ ±0.1), sign reversals are 13.7% of disagreements among the strong-4 observers (18.4% across all 6) |

Unanimous four-model classification covered 118 conversations (37 rising, 67 stable, 14 falling). Disagreements concentrate at the stable boundary, but sign reversals persist at roughly 1 in 7 disagreements even among strong observers — direction degrades more gracefully than position, not cleanly. With matched, chance-corrected metrics both position and direction are moderate; the case for the aggregate artifact is that it is the unit implementations actually rely on and must therefore be standardized and honestly reported, not that it is highly reliable. The protocol currently defines JSV (position), DV (step), and Trajectory (step sequence) — and stops one derivation short of that unit.

### Problem 2: `delegation_awareness` fails as a directly measured variable

DA is already rated observability **low** in the spec. Measured behavior is worse than "low":

- Same-model self-consistency: r=0.47 (lowest of the four dimensions)
- Inter-model agreement on direct DA: r=0.003 (gemma3-12b vs gpt-4.1-mini) — statistically indistinguishable from noise
- Observer-specific contamination: one major model assigned DA=10 to 24.4% of turns where two others assigned it to ≤0.5%

Deriving DA from the three behaviorally observable dimensions outperforms direct measurement:

```
DA = 0.162·JH + 0.570·CP + 0.268·ID     (normalized to 0–10)
```

- Small-model self-consistency: r=0.470 (direct) → r=0.645 (derived), +37%
- Inter-model agreement: r=0.003 (direct) → r=0.613 (derived)
- Stability: per-turn std 1.908 (direct) → 0.877 (derived consensus), −54%
- Nonlinear variants (full quadratic, R²=0.33 in-sample) underperform the linear form out-of-sample

The current structure — acknowledging low observability while requiring direct measurement — is internally inconsistent. The metacognition dimension is real; its measurement path should change.

### Problem 3: Task delegation contaminates judgment delegation

Both LLM and embedding observers misclassify task-passivity patterns ("continue please", bulk generation requests) as judgment delegation. The protocol already states that task requests are not judgment delegation (jsv-spec, judgment_holder note), but provides no structural way to mark which turns carry decision content at all. Filtering for decision-relevant turns is mentioned only as an implementation tip.

### Problem 4: Confidence labels are not comparable across observers

`confidence: low/medium/high` is nominal, observer-defined, and unusable for weighting or aggregation. Cross-observer consistency is flagged as unresolved in the spec; nothing in the record identifies *which* observer (model, prompt, version) produced a measurement, making the problem unanalyzable from the data alone.

### Problem 5: Current reporting conventions overstate measurement quality

A methodological self-audit of our own studies identified three practices that inflate apparent reliability, and that any implementation will reproduce unless the protocol standardizes reporting:

1. **Raw percent agreement without chance correction.** The originally-reported 65% trend-agreement figure sits over a 51.7% majority-class baseline (`stable` dominates); chance-corrected it is weighted κ≈0.51 — moderate, not high. Uncorrected percentages are not comparable across datasets with different base rates. (This correction has since been applied throughout this proposal; see the V3 Final Benchmark Report.)
2. **Single-run point estimates.** Observer-comparison deltas (e.g., DV correlation 0.53 vs 0.50) have been interpreted without confidence intervals; at n=300 conversations such differences may be within resampling noise.
3. **Undisclosed observer conditioning.** The reference labeling pipeline passes the observer its own prior-turn scores as context. This anchoring plausibly smooths trajectories — inflating trend coherence and deflating volatility — and is currently invisible in the output record.

None of these invalidate the findings, but a protocol that standardizes aggregate artifacts (Change 1) must also standardize how their quality is reported, or the artifacts travel with optimistic numbers attached.

## Proposed Changes

### Change 1: Interaction Summary as a fourth canonical artifact

```json
{
  "interaction_id": "session_001",
  "turn_count": 12,
  "decision_relevant_turn_count": 7,
  "dimensions": {
    "judgment_holder": {
      "trend": "rising",
      "slope": 0.42,
      "mean_level": 6.1,
      "volatility": 1.8,
      "peak_level": 9,
      "peak_turn": 10
    },
    "delegation_awareness": { "...": "same shape" },
    "cognitive_engagement": { "...": "same shape" },
    "information_seeking": { "...": "same shape" }
  },
  "basis": "all_turns | decision_relevant_turns"
}
```

Definitions (per dimension, computed over the JSV level sequence):

- **trend**: `rising | stable | falling`, from the sign of the least-squares slope with a dead zone: `|slope| ≤ θ → stable`. Default θ = 0.1 levels/turn (the empirically observed disagreement boundary); implementations may override but must record θ.
- **slope**: least-squares slope of level vs turn index.
- **mean_level**: arithmetic mean of levels.
- **volatility**: standard deviation of adjacent-turn deltas (i.e., of the DV components).
- **peak_level / peak_turn**: maximum level and the turn where it first occurs.

Constraints, consistent with existing artifacts:

- Purely descriptive; trend labels are directional, not evaluative. The existing prohibition on rankings and risk scores applies unchanged.
- Minimum 3 JSV snapshots to emit a summary (a slope over 2 points is a DV, not a trend).
- Null-holder turns are excluded from that dimension's aggregates; the count of excluded turns is recorded.
- The summary is **derivable**: given the JSV sequence and θ, the summary must be exactly recomputable. Validation tooling recomputes and compares, as it already does for trajectory continuity.

This also supersedes the informal pattern table in trajectory-spec (Gradual Delegation, Oscillation, …) with computable definitions: e.g. Oscillation = high volatility + stable trend; Reclamation = falling trend on judgment_holder.

### Change 2: Reclassify `delegation_awareness` as a derived observable

- JSV keeps the `delegation_awareness` field and its 0–10 semantics unchanged.
- The spec gains a **measurement-path** distinction: JH, CP, ID are *direct observables*; DA is a *derived observable* with derivation as the recommended default path.
- The reference derivation `DA = 0.162·JH + 0.570·CP + 0.268·ID` is published as an **informative annex** (versioned coefficients with training provenance), not a normative constant — coefficients are expected to be re-estimated as evidence accumulates.
- Direct DA measurement remains permitted; records must state the path: `"da_source": "derived" | "direct"`.
- The JH × DA interpretive matrix is unaffected structurally (it consumes DA values regardless of path).
- **Interpretive caveat (normative text)**: when `da_source` is `derived`, the matrix must be read as *behaviorally estimated* awareness — the "unconscious delegation" quadrant is partially a function of `cognitive_engagement` by construction. Claims about metacognition based on derived DA require at least one self-report validation in the implementation's deployment context (see Deferred: calibration standard).

### Change 3: Decision-relevant turn marker

JSV gains an optional field:

```json
"decision_relevant": true
```

Definition: the turn's human input engages an evaluative choice — selecting between options, assessing quality, accepting/rejecting a recommendation, or making a value-laden decision — as opposed to pure task execution requests, phatic turns, or mechanical continuations.

- Interaction Summary records which basis was used (`all_turns` vs `decision_relevant_turns`).
- Trajectories remain over all turns; filtering happens at the summary layer only, so trajectory continuity is untouched.
- Anchoring examples for the boundary (task vs judgment delegation) move from a spec footnote to a dedicated section with positive and negative examples.

### Change 4: Numeric uncertainty + observer profile

Confidence upgrade (backward compatible — the enum stays):

```json
"confidence": {
  "judgment_holder": "medium",
  "judgment_holder_numeric": { "std": 1.2, "n_observers": 3, "method": "ensemble_std" }
}
```

Observer profile block, required on JSV documents at the record level:

```json
"observer": {
  "id": "gemma4-26b/level_observer_3axis_cot/v1",
  "type": "llm | human | ensemble | embedding",
  "spec_version": "1.6.0"
}
```

Rationale: the documented Gemini DA contamination was only discoverable because observer identity happened to be tracked in our pipeline. The protocol should make cross-observer analysis possible from the canonical record itself. This is the data prerequisite for a future observer-calibration standard (see Deferred).

### Change 5: Measurement and reporting standards

Normative requirements for any published JDVP measurement or observer comparison:

1. **Chance-corrected agreement.** Observer agreement on categorical outputs (trend classes, derived categorical labels) is reported as weighted Cohen's κ or Krippendorff's α. Raw percent agreement may accompany these but never replace them.
2. **Resampled uncertainty.** Point estimates (correlations, agreement coefficients, summary statistics) carry bootstrap confidence intervals resampled at the **conversation** level, not the turn level — turns within a conversation are not independent observations.
3. **Conditioning disclosure.** JSV documents record whether the observer was conditioned on its own prior outputs:

   ```json
   "conditioning": "independent" | "prior_scores"
   ```

   Implementations should quantify the anchoring effect (label a subset both ways, compare volatility) before adopting `prior_scores` as a default. Interaction Summaries computed from `prior_scores` measurements inherit a smoothing bias in `volatility` and `trend` coherence.
4. **Ordinal robustness.** Levels are ordinal coordinate readings; interval arithmetic (least-squares slope, means) is a pragmatic approximation. Implementations must report the rank-based trend sign (Kendall τ) whenever it disagrees with the least-squares trend classification, and cross-conversation aggregation should prefer medians over means.

## Backward Compatibility

- All v1.5 documents remain valid: every addition is a new optional field or a new artifact type.
- `delegation_awareness` values are unchanged in type and range; `da_source` defaults to `"direct"` when absent (accurate for all pre-v1.6 data).
- Interaction Summary is additive; systems that only consume JSV/DV/Trajectory are unaffected.
- The confidence enum is retained; numeric uncertainty is an optional sibling.
- `conditioning` is optional; absence means undisclosed (accurate for all pre-v1.6 data). Reporting standards (Change 5) bind publications and comparisons, not stored documents — no existing artifact becomes invalid.

## Evidence Summary

| Claim | Evidence |
|---|---|
| Conversation trend is the operative unit | weighted κ≈0.51 [0.42–0.61] vs reference over a 51.7% majority baseline (raw 67.7%); 6-observer ordinal α=0.30; disagreements concentrated at slope ≈ ±0.1, sign reversals 13.7% of disagreements (strong-4) |
| Derived DA beats direct DA | self-consistency +37% (0.470→0.645); inter-model r 0.003→0.613; std −54% |
| Direct DA is observer-contaminated | DA=10 assigned to 24.4% of turns by one observer vs ≤0.5% by two others; removing it eliminated all DA>7 assignments outside jailbreak/roleplay content |
| Task/judgment confusion is systematic | embedding screening misclassifies "continue please" passivity as high delegation; fixed only by decision-relevance filtering |
| Linear derivation is the right form | quadratic R²=0.33 in-sample underperforms linear r=0.645 out-of-sample |
| Reporting practices inflate apparent quality | 67.7% raw trend agreement vs 51.7% majority-class baseline (weighted κ≈0.51); reference pipeline conditions the observer on its own prior-turn scores without recording it |

## Deferred (explicitly out of scope for v1.6)

1. **Observer calibration standard** (gold anchor set + agreement threshold for "compliant observer" status) — requires the observer profile block (Change 4) plus an expanded adjudicated anchor set; targeted for v1.7. Seed exists (102-item triple-annotated pilot).
2. **Cross-interaction (longitudinal) aggregation / trait estimation** — state→trait inference standard (test-retest, generalizability, minimum-observation rules); targeted for v1.7. Interaction Summary is its data prerequisite.
3. **Context-extension variables** — remain categorical; level-based measurement still not validated for domain extensions (unchanged from v1.5).

## Open Questions for Protocol Discussion

1. Should the stable dead zone θ=0.1 be normative (fixed) or a declared parameter with a recommended default? (This proposal: declared parameter, recorded in the artifact.)
2. Should Interaction Summary be computed over JSV levels (this proposal) or over DV sequences? Levels are more robust to single-turn spikes; DVs are the protocol's stated primary unit.
3. Should `decision_relevant` be boolean (this proposal) or a 0–10 weight? Boolean matches observed observer capability; a weight invites false precision.
4. Should derived-DA coefficients be frozen per spec minor version, or re-estimable with provenance? (This proposal: informative annex, versioned, re-estimable.)
5. Does trend labeling ("rising toward delegation") need explicit non-normative framing language to prevent downstream misuse as a risk score?
6. Should the least-squares slope remain the primary trend statistic (with Kendall τ as a disagreement check, per Change 5), or should the rank-based statistic be primary given that levels are ordinal?
7. Should conditioning the observer on its own prior-turn scores be prohibited outright rather than permitted-with-disclosure? Prohibition is cleaner for volatility semantics; disclosure preserves cheap sequential labeling pipelines. An ablation quantifying the anchoring effect should inform this before adoption.

## Implementation Notes

- Reference prompt: `config/prompts/level_observer_3axis_cot.txt` (3-axis CoT; "THINK FIRST, then score" + "who decides the DIRECTION, not who does the work").
- Recommended observer floor: gemma4:26b local (DV correlation 0.55 [0.51, 0.59], trend weighted κ 0.51 — statistically tied with sonnet-4/deepseek at $0) or gpt-4.1-class API.
- Summary computation is pure arithmetic over existing artifacts — no new extraction cost.
- POCv3 already implements trend classification and 4-model consensus; the schema work is codifying existing practice.
