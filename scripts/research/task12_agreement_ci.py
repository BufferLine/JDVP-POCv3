#!/usr/bin/env python3
"""Task 1 + 2: honest agreement re-reporting with conversation-level bootstrap CIs.

For every ShareGPT observer (vs the gpt41 reference) and every dimension, report
the three standard metrics with 95% CIs resampled at the CONVERSATION level:

  * trend agreement      : raw % (reproduces the historical figure) + weighted
                           Cohen's kappa + majority-class baseline
  * turn-level position   : Spearman rho on absolute levels
  * change vector (DV)     : Pearson r on adjacent-turn deltas

Also emits the multi-rater ordinal Krippendorff alpha per dimension and a
ranking-stability audit: any "model A > model B" claim whose 95% CIs overlap is
flagged for retraction.

Output: docs/research/v3-final/task12_results.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

import jdvp_v3_lib as L

REF = "gpt41"
OUT = Path(__file__).resolve().parents[2] / "docs/research/v3-final"
OUT.mkdir(parents=True, exist_ok=True)


# --- per-conversation unit builders (so the bootstrap can resample cleanly) ---

def trend_units(data, model, dim, theta=L.DEFAULT_THETA):
    """[(ref_trend, model_trend)] over conversations where both emit a trend."""
    tr_ref = L.conversation_trends(data[REF], dim, theta)
    tr_m = L.conversation_trends(data[model], dim, theta)
    common = set(tr_ref) & set(tr_m)
    return [(tr_ref[i], tr_m[i]) for i in sorted(common)]


def level_units(data, model, dim):
    """Per conversation: (ref_levels[], model_levels[]) aligned on shared turns."""
    units = []
    for iid in data[REF]:
        if iid not in data[model]:
            continue
        rt, mt = data[REF][iid], data[model][iid]
        shared = sorted(set(rt) & set(mt))
        rl, ml = [], []
        for tn in shared:
            rv, mv = rt[tn].get(dim), mt[tn].get(dim)
            if isinstance(rv, (int, float)) and isinstance(mv, (int, float)):
                rl.append(float(rv))
                ml.append(float(mv))
        if rl:
            units.append((rl, ml))
    return units


def dv_units(data, model, dim):
    """Per conversation: adjacent-turn deltas (ref_dv[], model_dv[])."""
    units = []
    for iid in data[REF]:
        if iid not in data[model]:
            continue
        rt, mt = data[REF][iid], data[model][iid]
        shared = sorted(set(rt) & set(mt))
        rl, ml = [], []
        for tn in shared:
            rv, mv = rt[tn].get(dim), mt[tn].get(dim)
            if isinstance(rv, (int, float)) and isinstance(mv, (int, float)):
                rl.append(float(rv)); ml.append(float(mv))
        if len(rl) >= 2:
            rdv = np.diff(rl); mdv = np.diff(ml)
            units.append((rdv, mdv))
    return units


# --- statistics over a list of units ---

def stat_weighted_kappa(units):
    if len(units) < 2:
        return np.nan
    a = [u[0] for u in units]
    b = [u[1] for u in units]
    return L.weighted_cohen_kappa(a, b)


def stat_raw_agreement(units):
    if not units:
        return np.nan
    return np.mean([u[0] == u[1] for u in units])


def stat_spearman(units):
    ref = np.concatenate([u[0] for u in units]) if units else np.array([])
    mod = np.concatenate([u[1] for u in units]) if units else np.array([])
    if len(ref) < 3:
        return np.nan
    return stats.spearmanr(ref, mod).statistic


def stat_pearson(units):
    ref = np.concatenate([u[0] for u in units]) if units else np.array([])
    mod = np.concatenate([u[1] for u in units]) if units else np.array([])
    if len(ref) < 3 or np.std(ref) == 0 or np.std(mod) == 0:
        return np.nan
    return stats.pearsonr(ref, mod).statistic


def ci(units, fn, n_boot=2000, seed=0):
    pt, lo, hi, _ = L.bootstrap_ci(units, fn, n_boot=n_boot, seed=seed)
    return {"point": _r(pt), "ci_lo": _r(lo), "ci_hi": _r(hi), "n": len(units)}


def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    data = L.load_all_sharegpt()
    others = [m for m in L.SHAREGPT_MODELS if m != REF]
    results = {"reference": REF, "theta": L.DEFAULT_THETA, "dimensions": {}}

    for dim in L.DIMS:
        short = L.DIM_SHORT[dim]
        dres = {"models": {}}

        # multi-rater ordinal Krippendorff alpha (all 6 observers) on trend
        trends = {m: L.conversation_trends(data[m], dim) for m in L.SHAREGPT_MODELS}
        common = set.intersection(*[set(trends[m]) for m in L.SHAREGPT_MODELS])
        rating_matrix = [{m: trends[m][i] for m in L.SHAREGPT_MODELS} for i in sorted(common)]
        dres["krippendorff_alpha_ordinal"] = _r(L.krippendorff_alpha_ordinal(rating_matrix))
        dres["n_common_convos"] = len(common)
        # pooled majority-class baseline across all observers' trends
        pooled = [trends[m][i] for i in common for m in L.SHAREGPT_MODELS]
        dres["majority_class_baseline"] = _r(L.majority_class_baseline(pooled))

        for m in others:
            tu = trend_units(data, m, dim)
            lu = level_units(data, m, dim)
            du = dv_units(data, m, dim)
            dres["models"][m] = {
                "trend_raw_agreement": ci(tu, stat_raw_agreement),
                "trend_weighted_kappa": ci(tu, stat_weighted_kappa),
                "level_spearman": ci(lu, stat_spearman),
                "dv_pearson": ci(du, stat_pearson),
            }
        results["dimensions"][dim] = dres

    # ranking-stability audit: for each (dim, metric), order models by point est,
    # flag adjacent pairs whose CIs overlap (claim not defensible)
    audit = []
    for dim in L.DIMS:
        for metric in ["trend_weighted_kappa", "level_spearman", "dv_pearson"]:
            rows = [(m, results["dimensions"][dim]["models"][m][metric]) for m in others]
            rows = [(m, r) for m, r in rows if r["point"] is not None]
            rows.sort(key=lambda t: t[1]["point"], reverse=True)
            for (m1, r1), (m2, r2) in zip(rows, rows[1:]):
                overlap = r1["ci_lo"] <= r2["ci_hi"] and r2["ci_lo"] <= r1["ci_hi"]
                audit.append({
                    "dimension": L.DIM_SHORT[dim], "metric": metric,
                    "higher": m1, "lower": m2,
                    "higher_pt": r1["point"], "lower_pt": r2["point"],
                    "ci_overlap": bool(overlap),
                    "claim_defensible": not overlap,
                })
    results["ranking_audit"] = audit

    (OUT / "task12_results.json").write_text(json.dumps(results, indent=2))
    # console summary
    print(f"Reference observer: {REF}  theta={L.DEFAULT_THETA}\n")
    for dim in L.DIMS:
        d = results["dimensions"][dim]
        print(f"== {L.DIM_SHORT[dim]} ==  Kripp-alpha(6)={d['krippendorff_alpha_ordinal']} "
              f"majority-baseline={d['majority_class_baseline']}")
        print(f"  {'model':12s} {'trend raw':>18s} {'trend wkappa':>20s} {'level rho':>18s} {'DV r':>18s}")
        for m in others:
            r = d["models"][m]
            def f(x): return f"{x['point']} [{x['ci_lo']},{x['ci_hi']}]"
            print(f"  {m:12s} {f(r['trend_raw_agreement']):>18s} {f(r['trend_weighted_kappa']):>20s} "
                  f"{f(r['level_spearman']):>18s} {f(r['dv_pearson']):>18s}")
        print()
    n_overlap = sum(1 for a in audit if a["ci_overlap"])
    print(f"Ranking audit: {n_overlap}/{len(audit)} adjacent-pair claims have overlapping CIs "
          f"(not defensible).")


if __name__ == "__main__":
    main()
