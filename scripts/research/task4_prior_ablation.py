#!/usr/bin/env python3
"""Task 4: prior-anchoring ablation analysis.

Compares the same conversations labelled by gemma4:26b under two regimes:
  prior_scores  — observer sees its own previous-turn JD/CP/ID (reference pipeline)
  independent   — no prior-score block

Hypothesis (v1.6 Change 5.3): feeding the observer its own prior scores SMOOTHS
trajectories -> lower per-conversation volatility and higher trend coherence.

Per conversation x dimension we compute:
  volatility        = std of adjacent-turn deltas (v1.6 definition)
  mean_abs_delta    = mean |adjacent delta|
  lag1_autocorr     = autocorrelation of the level sequence (smoothness)
  trend label       = LS slope with dead zone theta=0.1
Then paired (by conversation) comparisons + Wilcoxon signed-rank, plus how often
the trend label flips between regimes.

Output: docs/research/v3-final/task4_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import jdvp_v3_lib as L

OUT = Path(__file__).resolve().parents[2] / "docs/research/v3-final"
DIMS = ["judgment_delegation", "cognitive_passivity", "information_dependency",
        "delegation_awareness_derived"]


def load_ablation(cond):
    root = L.SILVER / f"sharegpt-3axis-ablation-{cond}"
    out = {}
    if not root.exists():
        return out
    for iid_dir in sorted(root.iterdir()):
        if not iid_dir.is_dir():
            continue
        turns = {}
        for tf in iid_dir.glob("turn-*.json"):
            tn = int(tf.stem.split("-")[1])
            try:
                turns[tn] = json.loads(tf.read_text())
            except json.JSONDecodeError:
                continue
        if turns:
            out[iid_dir.name] = turns
    return out


def seq(turns, dim):
    xs, ys = L.level_sequence(turns, dim)
    return xs, ys


def volatility(ys):
    return float(np.std(np.diff(ys))) if len(ys) >= 2 else np.nan


def mean_abs_delta(ys):
    return float(np.mean(np.abs(np.diff(ys)))) if len(ys) >= 2 else np.nan


def lag1_autocorr(ys):
    if len(ys) < 3 or np.std(ys) == 0:
        return np.nan
    a, b = ys[:-1], ys[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    prior = load_ablation("prior_scores")
    indep = load_ablation("independent")
    paired = sorted(set(prior) & set(indep))
    results = {"n_prior": len(prior), "n_independent": len(indep),
               "n_paired": len(paired), "paired_iids": paired, "dimensions": {}}

    if not paired:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "task4_results.json").write_text(json.dumps(results, indent=2))
        print("No paired conversations yet — labelling still in progress.")
        print(f"prior={len(prior)} independent={len(indep)}")
        return

    for dim in DIMS:
        pv, iv, pmad, imad, pac, iac = [], [], [], [], [], []
        ptr, itr = [], []
        for iid in paired:
            _, yp = seq(prior[iid], dim)
            _, yi = seq(indep[iid], dim)
            if len(yp) < 3 or len(yi) < 3:
                continue
            pv.append(volatility(yp)); iv.append(volatility(yi))
            pmad.append(mean_abs_delta(yp)); imad.append(mean_abs_delta(yi))
            pac.append(lag1_autocorr(yp)); iac.append(lag1_autocorr(yi))
            xp, _ = seq(prior[iid], dim); xi, _ = seq(indep[iid], dim)
            ptr.append(L.classify_trend(L.ls_slope(xp, yp)))
            itr.append(L.classify_trend(L.ls_slope(xi, yi)))

        pv, iv = np.array(pv), np.array(iv)

        def paired_stat(a, b):
            a, b = np.array(a), np.array(b)
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            if len(a) < 3:
                return {"n": int(len(a))}
            try:
                w = stats.wilcoxon(a, b, zero_method="wilcox")
                p = float(w.pvalue)
            except ValueError:
                p = None
            return {"n": int(len(a)),
                    "prior_mean": round(float(a.mean()), 4),
                    "independent_mean": round(float(b.mean()), 4),
                    "mean_diff_prior_minus_indep": round(float((a - b).mean()), 4),
                    "wilcoxon_p": None if p is None else round(p, 4)}

        trend_flips = sum(1 for x, y in zip(ptr, itr) if x != y and x and y)
        n_tr = sum(1 for x, y in zip(ptr, itr) if x and y)
        results["dimensions"][L.DIM_SHORT[dim]] = {
            "volatility": paired_stat(pv, iv),
            "mean_abs_delta": paired_stat(pmad, imad),
            "lag1_autocorr": paired_stat(pac, iac),
            "trend_flip_rate": round(trend_flips / n_tr, 4) if n_tr else None,
            "trend_flips": trend_flips, "n_trend_pairs": n_tr,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "task4_results.json").write_text(json.dumps(results, indent=2))

    print(f"Paired conversations: {len(paired)}\n")
    print(f"{'dim':4s}{'metric':16s}{'prior':>9s}{'indep':>9s}{'diff':>9s}{'wilcox_p':>10s}")
    for short, d in results["dimensions"].items():
        for metric in ["volatility", "mean_abs_delta", "lag1_autocorr"]:
            s = d[metric]
            if "prior_mean" in s:
                print(f"{short:4s}{metric:16s}{s['prior_mean']:>9.3f}{s['independent_mean']:>9.3f}"
                      f"{s['mean_diff_prior_minus_indep']:>9.3f}{str(s['wilcoxon_p']):>10s}")
        print(f"     trend_flip_rate = {d['trend_flip_rate']} ({d['trend_flips']}/{d['n_trend_pairs']})")
        print()
    print("Hypothesis: prior anchoring smooths -> prior volatility < independent "
          "(negative diff), prior autocorr > independent (positive diff).")


if __name__ == "__main__":
    main()
