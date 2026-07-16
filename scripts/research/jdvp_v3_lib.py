#!/usr/bin/env python3
"""Shared library for the V3 final re-analysis (Tasks 1-5).

Loads the ShareGPT 6-model 3-axis+derived-DA silver labels and the WildChat
numeric 4-axis labels, and provides the standard estimators used across the
report: least-squares / Theil-Sen trend classification with a dead zone,
weighted Cohen's kappa, ordinal Krippendorff's alpha, Spearman rho, and
conversation-level bootstrap confidence intervals.

Data lives in the primary checkout (data/ is gitignored → absent in worktrees).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

# Repo root by default (…/scripts/research/ -> parents[2]). data/ is gitignored,
# so in a git worktree it is absent; point JDVP_DATA_ROOT at the primary checkout
# (which holds the extracted archives) to reproduce there.
DATA_ROOT = Path(os.environ.get("JDVP_DATA_ROOT", Path(__file__).resolve().parents[2]))
SILVER = DATA_ROOT / "data/silver"

# The six ShareGPT observers with (near) full 4,103-turn coverage.
SHAREGPT_MODELS = ["gpt41", "sonnet", "deepseek", "nemotron", "gpt54nano", "gemma4-26b"]

DIMS = ["judgment_delegation", "cognitive_passivity", "information_dependency",
        "delegation_awareness_derived"]
DIM_SHORT = {"judgment_delegation": "JH", "cognitive_passivity": "CP",
             "information_dependency": "ID", "delegation_awareness_derived": "DA"}

TREND_ORDER = {"falling": -1, "stable": 0, "rising": 1}
DEFAULT_THETA = 0.1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sharegpt_model(model: str) -> dict:
    """Return {iid: {turn_number: {dim: value}}} for one ShareGPT observer."""
    root = SILVER / f"sharegpt-3axis-{model}-cot"
    out: dict[str, dict[int, dict]] = {}
    for iid_dir in sorted(root.iterdir()):
        if not iid_dir.is_dir():
            continue
        turns: dict[int, dict] = {}
        for tf in iid_dir.glob("turn-*.json"):
            tn = int(tf.stem.split("-")[1])
            try:
                d = json.loads(tf.read_text())
            except json.JSONDecodeError:
                continue
            turns[tn] = d
        if turns:
            out[iid_dir.name] = turns
    return out


def load_all_sharegpt() -> dict:
    """{model: {iid: {turn: {dim: val}}}} for all six observers."""
    return {m: load_sharegpt_model(m) for m in SHAREGPT_MODELS}


def level_sequence(turns: dict, dim: str):
    """Ordered (turn_index, level) arrays for one conversation/dimension.

    turn_index is the actual turn_number (handles missing turns in
    deepseek/nemotron), level is the observer's value. Null/missing skipped.
    """
    xs, ys = [], []
    for tn in sorted(turns):
        v = turns[tn].get(dim)
        if isinstance(v, (int, float)):
            xs.append(tn)
            ys.append(float(v))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

def ls_slope(xs, ys):
    if len(xs) < 2:
        return np.nan
    return np.polyfit(xs, ys, 1)[0]


def theil_sen_slope(xs, ys):
    if len(xs) < 2:
        return np.nan
    return stats.theilslopes(ys, xs)[0]


def classify_trend(slope, theta=DEFAULT_THETA):
    if not np.isfinite(slope):
        return None
    if slope > theta:
        return "rising"
    if slope < -theta:
        return "falling"
    return "stable"


def conversation_trends(model_data: dict, dim: str, theta=DEFAULT_THETA,
                        min_snapshots=3, method="ls"):
    """{iid: trend_label} for every conversation with >= min_snapshots levels."""
    slope_fn = ls_slope if method == "ls" else theil_sen_slope
    out = {}
    for iid, turns in model_data.items():
        xs, ys = level_sequence(turns, dim)
        if len(ys) < min_snapshots:
            continue
        out[iid] = classify_trend(slope_fn(xs, ys), theta)
    return out


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

def weighted_cohen_kappa(a, b, categories=("falling", "stable", "rising"),
                         weights="linear"):
    """Weighted Cohen's kappa for two ordinal raters over aligned label lists."""
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    a = [idx[x] for x in a]
    b = [idx[x] for x in b]
    n = len(a)
    if n == 0:
        return np.nan
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= n
    ra = O.sum(axis=1)
    cb = O.sum(axis=0)
    E = np.outer(ra, cb)
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                W[i, j] = abs(i - j) / (k - 1)
            else:  # quadratic
                W[i, j] = ((i - j) / (k - 1)) ** 2
    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return np.nan
    return 1 - num / den


def krippendorff_alpha_ordinal(rating_matrix, categories=("falling", "stable", "rising")):
    """Ordinal Krippendorff's alpha via the coincidence-matrix formulation
    (Krippendorff 2011). rating_matrix: list of dicts {rater: label} per unit;
    missing raters allowed. `categories` is the ordered value set.

    Validated: perfect agreement -> 1.0; independent labels -> ~0.
    """
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # coincidence matrix o: each within-unit ordered pair contributes 1/(m_u-1)
    o = np.zeros((k, k))
    for unit in rating_matrix:
        vals = [idx[v] for v in unit.values()]
        m = len(vals)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[vals[i], vals[j]] += w
    n_c = o.sum(axis=1)          # coincidence marginals
    n = n_c.sum()
    if n < 2:
        return np.nan

    def delta(c, d):
        lo, hi = (c, d) if c <= d else (d, c)
        s = n_c[lo:hi + 1].sum() - (n_c[c] + n_c[d]) / 2.0
        return s ** 2

    Do = sum(o[c, d] * delta(c, d) for c in range(k) for d in range(k)) / n
    De = sum(n_c[c] * n_c[d] * delta(c, d) for c in range(k) for d in range(k)) / (n * (n - 1))
    if De == 0:
        return np.nan
    return 1 - Do / De


def majority_class_baseline(labels):
    """Accuracy of always predicting the modal class."""
    labels = list(labels)
    if not labels:
        return np.nan
    vals, counts = np.unique(labels, return_counts=True)
    return counts.max() / len(labels)


# ---------------------------------------------------------------------------
# Bootstrap (conversation-level resampling)
# ---------------------------------------------------------------------------

def bootstrap_ci(units, stat_fn, n_boot=2000, alpha=0.05, seed=0):
    """Conversation-level bootstrap CI. `units` is a list (one entry per
    conversation); stat_fn maps a resampled list of units -> scalar."""
    rng = np.random.default_rng(seed)
    units = list(units)
    n = len(units)
    point = stat_fn(units)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = [units[i] for i in rng.integers(0, n, n)]
        boots[b] = stat_fn(sample)
    boots = boots[np.isfinite(boots)]
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return point, lo, hi, boots


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return stats.spearmanr(x[m], y[m]).statistic
