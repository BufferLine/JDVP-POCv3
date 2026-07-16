# V3 Final Benchmark — re-analysis scripts

Honest re-reporting of the V3 extended labeling assets, backing
[`docs/research/2026-07-16-v3-final-benchmark-report.md`](../../docs/research/2026-07-16-v3-final-benchmark-report.md)
and the [4-axis construct codebook](../../docs/research/2026-07-16-jdvp-4axis-construct-codebook.md).
Implements the measurement/reporting standards of JDVP v1.6 Change 5
(chance-corrected agreement, conversation-level bootstrap CIs, conditioning
ablation, ordinal robustness).

## Data location

`data/` is gitignored, so it is **absent inside git worktrees**. Point the
scripts at the checkout that holds the extracted archives:

```bash
export JDVP_DATA_ROOT=/path/to/JDVP-POCv3   # checkout whose data/silver/* is extracted
```

Without the env var the scripts default to the repo root of this checkout.
Required assets (extract from `data/silver/*.tar.gz` if only archives are present):
`sharegpt-3axis-{gpt41,sonnet,deepseek,nemotron,gpt54nano,gemma4-26b}-cot/`,
`wildchat-level-experiment/`, `level-extreme-test-level-experiment/`
(the last two live inside `level-experiment-archive-20260403.tar.gz`; `task3` reads
that archive directly).

## Run

```bash
python3 task12_agreement_ci.py    # Task 1+2: trend kappa / Spearman / DV, conversation bootstrap CIs
python3 task_f1_detection.py      # 3rd standard metric: high-delegation (level>=7) detection F1
python3 task3_da_crosscorpus.py   # Task 3: DA-coefficient cross-corpus validation
python3 task4_prior_ablation.py   # Task 4: prior-anchoring ablation analysis (needs the ablation labels)
```

Results are written to `docs/research/v3-final/*.json`. `jdvp_v3_lib.py` holds the
shared loaders and estimators (trend classification, weighted Cohen's kappa,
ordinal Krippendorff alpha, conversation-level bootstrap).

## Regenerating the ablation labels (Task 4)

```bash
# gemma4:26b via local ollama; ~12 s/turn
python3 ablation_label.py --conditioning prior_scores --subset pilot15
python3 ablation_label.py --conditioning independent  --subset pilot15
```

Subsets are defined in `ablation_iids.json` (`pilot15`, `all30`).
