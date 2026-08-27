# Human-AI Calibration Dataset

This dataset establishes measurement reliability; it does not establish a critical-thinking score.

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

The report provides pairwise human agreement, observer exact agreement and mean absolute error on adjudicated rows, plus any recorded per-turn latency and cost estimates. It must be read as calibration evidence, not a psychological assessment result.
