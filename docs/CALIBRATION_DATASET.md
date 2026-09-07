# Human-AI Calibration Dataset

This legacy Human–AI dataset contract supports measurement evaluation; the presence of a dataset does not establish reliability or a critical-thinking score. Human-expert annotation is no longer a required stage of the planned automated research loop.

The current loader still requires `measurement_profile: "human_ai"` and `schema_version: "jdvp-human-calibration-v1"`; do not rename these wire values or claim other profile support without a code migration. Existing human annotations may be preserved, while AI-reference sources must be identified explicitly (for example annotator `kind: "ai_reference"` plus model/prompt versions in provenance). `adjudicated_labels` means an explicitly resolved target, not necessarily a human judgment. Synthetic demo rows remain synthetic. The future loop must persist independent labels, challenge results, abstentions, sampling policy, and framework versions in a separately versioned contract. See [AUTOMATED_RESEARCH_LOOP.md](./AUTOMATED_RESEARCH_LOOP.md).

## Current Loader Limitation

The loader validates `annotator_id` and labels, but does not validate or use annotator `kind` to separate sources. The report pools all annotation pairs into `annotator_agreement`; human and AI annotations can therefore be mixed in the same aggregate. Provenance fields above are documentation requirements, not currently enforced source-aware evaluation. Until source-stratified reporting is implemented, evaluate homogeneous source cohorts in separate datasets/runs and label their reports accordingly. A mixed-source aggregate must not be presented as human-only agreement or independent AI-reference agreement. Source validation and stratified reporting belong to the R1 contract migration.

## Layout

```text
<dataset-root>/
├── manifest.json
└── annotations.jsonl
```

`manifest.json` must declare `schema_version: "jdvp-human-calibration-v1"`, `measurement_profile: "human_ai"`, a non-empty `consent_basis`, the intended research use, protocol version, and pseudonymous annotator identifiers.

Each `annotations.jsonl` row is one interaction turn:

```json
{
  "interaction_id": "example-001",
  "turn_number": 0,
  "annotations": [
    {"annotator_id": "a-01", "labels": {"judgment_holder": 2, "delegation_awareness": 2, "cognitive_engagement": 2, "information_seeking": 2}},
    {"annotator_id": "a-02", "labels": {"judgment_holder": 3, "delegation_awareness": 2, "cognitive_engagement": 2, "information_seeking": 2}}
  ],
  "adjudicated_labels": {"judgment_holder": 2, "delegation_awareness": 2, "cognitive_engagement": 2, "information_seeking": 2}
}
```

Only explicit `adjudicated_labels` are used as the observer target. Disagreement is reported; it is never silently converted into a majority-vote gold label.

## Evaluation

Run an extraction against the same interactions, then evaluate its stored run:

```bash
python3 scripts/evaluate_calibration.py \
  --dataset-root path/to/calibration-set \
  --run-dir data/runs/<run-id> \
  --output data/runs/<run-id>/calibration_report.json
```

The report provides pairwise annotator agreement (interpreted according to the declared label source), observer exact agreement and mean absolute error on adjudicated rows, plus any recorded per-turn latency and cost estimates. It must be read as calibration evidence, not a psychological assessment result.

## Checked-in synthetic demo

`data/calibration/demo-human-ai-v1/` is a fully synthetic three-turn demo paired with `data/fixtures/sample_interaction.json`. It is included to prove the dataset and reporting path end to end. It contains no human annotations and must not be reported as human calibration evidence.

Run the deterministic fixture observer, then generate its report:

```bash
JDVP_ESTIMATED_COST_USD_PER_TURN=0 python3 -m src.pipeline.run_poc \
  --input data/fixtures/sample_interaction.json \
  --run-id calibration-demo-fixture-v1 \
  --track fixture_hint

python3 scripts/evaluate_calibration.py \
  --dataset-root data/calibration/demo-human-ai-v1 \
  --run-dir data/runs/calibration-demo-fixture-v1 \
  --output data/runs/calibration-demo-fixture-v1/calibration_report.json
```

To exercise the intended low-cost observer path, use a complete local embedding model and an explicitly supplied prototype pack:

```bash
export JDVP_EMBEDDING_MODEL_PATH=/absolute/path/to/complete/embedding-model
export JDVP_EMBEDDING_PROTOTYPE_PACK_PATH=data/fewshot/general-fixture-pack-v1.json

JDVP_ESTIMATED_COST_USD_PER_TURN=0 python3 -m src.pipeline.run_poc \
  --input data/fixtures/sample_interaction.json \
  --run-id calibration-demo-embedding-v1 \
  --track embedding_screen
```

The checked-in prototype pack contains the same fixture turns as this demo. That makes this a wiring smoke test only: it must not be read as an out-of-sample embedding result.
