#!/usr/bin/env python3
"""Third standard metric: high-delegation turn-detection F1 (vs gpt41 reference).

The completion criterion asks every model to be re-reported on three standard
metrics: Spearman rho (position, task12), weighted kappa (trend, task12), and
F1. F1 needs a categorical target, so we frame the product-relevant detection
task: does the observer flag the same "high" turns as the gpt41 reference?

  positive class: level >= 7 on a dimension (a strong-delegation / strong-
  passivity / strong-dependency / low-awareness moment)

Reported per model x dimension with conversation-level bootstrap 95% CI, plus
the reference positive base rate (so F1 is read against prevalence).

Output: docs/research/v3-final/task_f1_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import jdvp_v3_lib as L

REF = "gpt41"
THRESH = 7.0
OUT = Path(__file__).resolve().parents[2] / "docs/research/v3-final"


def f1_units(data, model, dim):
    """Per conversation: (ref_bin[], model_bin[]) on shared turns."""
    units = []
    for iid in data[REF]:
        if iid not in data[model]:
            continue
        rt, mt = data[REF][iid], data[model][iid]
        rb, mb = [], []
        for tn in sorted(set(rt) & set(mt)):
            rv, mv = rt[tn].get(dim), mt[tn].get(dim)
            if isinstance(rv, (int, float)) and isinstance(mv, (int, float)):
                rb.append(rv >= THRESH); mb.append(mv >= THRESH)
        if rb:
            units.append((np.array(rb), np.array(mb)))
    return units


def f1_from_units(units):
    if not units:
        return np.nan
    ref = np.concatenate([u[0] for u in units])
    mod = np.concatenate([u[1] for u in units])
    tp = np.sum(ref & mod); fp = np.sum(~ref & mod); fn = np.sum(ref & ~mod)
    if tp == 0:
        return 0.0 if (fp + fn) > 0 else np.nan
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def base_rate(units):
    ref = np.concatenate([u[0] for u in units]) if units else np.array([])
    return float(ref.mean()) if len(ref) else np.nan


def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    data = L.load_all_sharegpt()
    others = [m for m in L.SHAREGPT_MODELS if m != REF]
    res = {"reference": REF, "threshold": THRESH, "positive_class": "level>=7",
           "dimensions": {}}
    for dim in L.DIMS:
        dd = {"models": {}}
        # reference base rate on this dim (over all ref turns)
        allref = []
        for iid, turns in data[REF].items():
            for tn, t in turns.items():
                v = t.get(dim)
                if isinstance(v, (int, float)):
                    allref.append(v >= THRESH)
        dd["reference_base_rate"] = _r(np.mean(allref))
        for m in others:
            units = f1_units(data, m, dim)
            pt, lo, hi, _ = L.bootstrap_ci(units, f1_from_units, n_boot=2000)
            dd["models"][m] = {"f1": _r(pt), "ci_lo": _r(lo), "ci_hi": _r(hi),
                               "n_convos": len(units)}
        res["dimensions"][dim] = dd
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "task_f1_results.json").write_text(json.dumps(res, indent=2))

    print(f"High-delegation detection F1 (level>={THRESH:.0f}, vs {REF}), "
          f"conversation-bootstrap 95% CI\n")
    for dim in L.DIMS:
        dd = res["dimensions"][dim]
        print(f"== {L.DIM_SHORT[dim]} ==  ref base-rate(high)={dd['reference_base_rate']}")
        for m in others:
            r = dd["models"][m]
            print(f"  {m:12s} F1={r['f1']} [{r['ci_lo']},{r['ci_hi']}]")
        print()


if __name__ == "__main__":
    main()
