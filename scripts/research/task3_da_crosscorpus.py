#!/usr/bin/env python3
"""Task 3: cross-corpus validation of the derived-DA coefficients.

PROVENANCE CORRECTION (documented in the report):
  The published coefficients DA = 0.162*JH + 0.570*CP + 0.268*ID were fit on the
  WildChat numeric multi-observer labels (derived-da-study Exp 2), NOT on
  ShareGPT. ShareGPT carries no *direct* DA at all, so it cannot be the fit
  corpus. An honest generalization test therefore is:

    (A) Published-formula transfer  : apply the frozen coefficients to each
        observer's direct DA, report R^2 / MAE (fit corpus = WildChat; OOD = extreme-test).
    (B) Within-WildChat honest CV   : 5-fold BY CONVERSATION, refit OLS on train
        folds, score held-out folds (guards against in-sample optimism), and
        compare refit coefficients to the published ones.
    (C) True cross-corpus hold-out  : fit on ALL WildChat, predict the
        extreme-test corpus (different distribution), report R^2 / MAE.

Each observer's own JH/CP/ID reconstruct its own direct DA (within-observer), so
the test isolates the functional form from cross-observer disagreement.

Reads the numeric labels straight from the committed archives in data/silver/.
Output: docs/research/v3-final/task3_results.json
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
from scipy import stats

import jdvp_v3_lib as L

OUT = Path(__file__).resolve().parents[2] / "docs/research/v3-final"
OUT.mkdir(parents=True, exist_ok=True)

PUBLISHED = {"judgment_delegation": 0.162, "cognitive_passivity": 0.570,
             "information_dependency": 0.268}
AXES = ["judgment_delegation", "cognitive_passivity", "information_dependency"]
TARGET = "delegation_awareness"

ARCHIVE = L.SILVER / "level-experiment-archive-20260403.tar.gz"
WC_PREFIX = "data/silver/wildchat-level-experiment/"
ET_PREFIX = "data/silver/level-extreme-test-level-experiment/"

# Headline observers: capable, no documented DA contamination. gemini* is
# reported separately (derived-da-study Exp 3: DA=10 on 24.4% of turns).
HEADLINE = ["openrouter-gpt41", "openrouter-sonnet4", "openrouter-haiku",
            "openrouter-gpt41mini", "openrouter-gpt41nano", "gemma3-12b"]
CONTAMINATED = ["openrouter-gemini25flash", "gemini-flash", "gemini"]


def load_corpus(prefix):
    """{model: {iid: [ {axis:val,...}, ... ]}} of turns with all 4 axes present."""
    out: dict = {}
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]           # model/iid/turn-N.json
            parts = rel.split("/")
            if len(parts) != 3 or not parts[2].startswith("turn-"):
                continue
            model, iid, _ = parts
            try:
                d = json.loads(tf.extractfile(member).read())
            except Exception:
                continue
            if all(isinstance(d.get(k), (int, float)) for k in AXES + [TARGET]):
                out.setdefault(model, {}).setdefault(iid, []).append(
                    {k: float(d[k]) for k in AXES + [TARGET]})
    return out


def matrix(model_convs, iids=None):
    """Stack turns -> (X[JH,CP,ID], y[DA], conv_index[]) for given conversations."""
    if iids is None:
        iids = list(model_convs)
    X, y, ci = [], [], []
    for c_idx, iid in enumerate(iids):
        for t in model_convs.get(iid, []):
            X.append([t[a] for a in AXES]); y.append(t[TARGET]); ci.append(c_idx)
    return np.array(X), np.array(y), np.array(ci)


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def pearson(y, yhat):
    if np.std(y) == 0 or np.std(yhat) == 0:
        return np.nan
    return float(stats.pearsonr(y, yhat).statistic)


def apply_published(X):
    w = np.array([PUBLISHED[a] for a in AXES])
    return np.clip(X @ w, 0, 10)


def ols_no_intercept(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def kfold_by_conversation(model_convs, k=5, seed=0):
    """Out-of-fold predictions refitting OLS(no intercept) on train convos."""
    iids = list(model_convs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(iids))
    folds = np.array_split(order, k)
    y_all, yhat_all, coefs = [], [], []
    for f in range(k):
        test_idx = set(folds[f].tolist())
        train_iids = [iids[i] for i in range(len(iids)) if i not in test_idx]
        test_iids = [iids[i] for i in folds[f]]
        Xtr, ytr, _ = matrix(model_convs, train_iids)
        Xte, yte, _ = matrix(model_convs, test_iids)
        if len(ytr) < 4 or len(yte) == 0:
            continue
        w = ols_no_intercept(Xtr, ytr)
        coefs.append(w)
        y_all.append(yte); yhat_all.append(np.clip(Xte @ w, 0, 10))
    y_all = np.concatenate(y_all); yhat_all = np.concatenate(yhat_all)
    return y_all, yhat_all, np.mean(coefs, axis=0)


def boot_metric(model_convs, metric_fn, predictor, n_boot=1000, seed=1):
    """Conversation-level bootstrap of a metric over (predict->score)."""
    iids = list(model_convs)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        samp = [iids[i] for i in rng.integers(0, len(iids), len(iids))]
        X, y, _ = matrix(model_convs, samp)
        if len(y) < 3:
            continue
        vals.append(metric_fn(y, predictor(X)))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    wc = load_corpus(WC_PREFIX)
    et = load_corpus(ET_PREFIX)
    results = {"published_coefficients": PUBLISHED, "observers": {}}

    all_models = HEADLINE + [m for m in CONTAMINATED if m in wc]
    for model in all_models:
        if model not in wc:
            continue
        wc_m = wc[model]
        entry = {"contaminated": model in CONTAMINATED,
                 "wildchat": {"n_convos": len(wc_m)}}

        # (A) published-formula transfer on WildChat
        Xw, yw, _ = matrix(wc_m)
        yhat = apply_published(Xw)
        lo, hi = boot_metric(wc_m, r2, apply_published)
        entry["wildchat"]["published_r2"] = _r(r2(yw, yhat))
        entry["wildchat"]["published_r2_ci"] = [_r(lo), _r(hi)]
        entry["wildchat"]["published_mae"] = _r(mae(yw, yhat))
        entry["wildchat"]["published_pearson"] = _r(pearson(yw, yhat))
        entry["wildchat"]["da_direct_mean"] = _r(yw.mean())
        entry["wildchat"]["da_direct_std"] = _r(yw.std())

        # (B) within-WildChat honest 5-fold CV
        y_oof, yhat_oof, refit_w = kfold_by_conversation(wc_m)
        entry["wildchat"]["cv_r2"] = _r(r2(y_oof, yhat_oof))
        entry["wildchat"]["cv_mae"] = _r(mae(y_oof, yhat_oof))
        entry["wildchat"]["refit_coefficients"] = {a: _r(w) for a, w in zip(AXES, refit_w)}

        # (C) true cross-corpus: fit on all WildChat, predict extreme-test
        if model in et:
            et_m = et[model]
            w_full = ols_no_intercept(Xw, yw)
            Xe, ye, _ = matrix(et_m)
            yhat_pub = apply_published(Xe)
            yhat_fit = np.clip(Xe @ w_full, 0, 10)
            lo_p, hi_p = boot_metric(et_m, r2, apply_published)
            entry["extreme_test"] = {
                "n_convos": len(et_m),
                "da_direct_mean": _r(ye.mean()), "da_direct_std": _r(ye.std()),
                "published_r2": _r(r2(ye, yhat_pub)), "published_r2_ci": [_r(lo_p), _r(hi_p)],
                "published_mae": _r(mae(ye, yhat_pub)),
                "published_pearson": _r(pearson(ye, yhat_pub)),
                "wildchat_fit_r2": _r(r2(ye, yhat_fit)),
                "wildchat_fit_mae": _r(mae(ye, yhat_fit)),
            }
        results["observers"][model] = entry

    # pooled headline (concatenate clean observers, resample by conversation)
    pooled = {}
    for model in HEADLINE:
        for iid, turns in wc.get(model, {}).items():
            pooled[f"{model}:{iid}"] = turns
    Xp, yp, _ = matrix(pooled)
    results["pooled_headline"] = {
        "n_observer_convos": len(pooled),
        "published_r2": _r(r2(yp, apply_published(Xp))),
        "published_mae": _r(mae(yp, apply_published(Xp))),
        "published_pearson": _r(pearson(yp, apply_published(Xp))),
        "da_direct_std": _r(yp.std()),
    }
    yq, yhq, wq = kfold_by_conversation(pooled)
    results["pooled_headline"]["cv_r2"] = _r(r2(yq, yhq))
    results["pooled_headline"]["refit_coefficients"] = {a: _r(w) for a, w in zip(AXES, wq)}

    (OUT / "task3_results.json").write_text(json.dumps(results, indent=2))

    # console
    print("Published DA = 0.162*JH + 0.570*CP + 0.268*ID   (fit corpus: WildChat)")
    print("R2 is NOT comparable across corpora (WildChat DA is low-variance, "
          "extreme-test spans 0-10); read MAE + Pearson r for transfer.\n")
    print(f"{'observer':26s}{'WCpubR2':>9s}{'WC r':>7s}{'WCcvR2':>8s}{'WC MAE':>8s}"
          f"{'ETpubR2':>9s}{'ET r':>7s}{'ET MAE':>8s}  refit(JH,CP,ID)")
    for m, e in results["observers"].items():
        w = e["wildchat"]
        et_e = e.get("extreme_test", {})
        rc = e["wildchat"]["refit_coefficients"]
        tag = " *CONTAM" if e["contaminated"] else ""
        print(f"{m:26s}{str(w['published_r2']):>9s}{str(w['published_pearson']):>7s}"
              f"{str(w['cv_r2']):>8s}{str(w['published_mae']):>8s}"
              f"{str(et_e.get('published_r2')):>9s}{str(et_e.get('published_pearson')):>7s}"
              f"{str(et_e.get('published_mae')):>8s}"
              f"  ({rc['judgment_delegation']},{rc['cognitive_passivity']},{rc['information_dependency']}){tag}")
    ph = results["pooled_headline"]
    print(f"\nPooled headline (clean observers): pub R2={ph['published_r2']} "
          f"pub r={ph['published_pearson']} cv R2={ph['cv_r2']} MAE={ph['published_mae']}")
    print(f"  refit coeffs={ph['refit_coefficients']}  vs published {PUBLISHED}")


if __name__ == "__main__":
    main()
