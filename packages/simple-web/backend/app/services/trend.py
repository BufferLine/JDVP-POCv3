from __future__ import annotations

from collections import Counter

import numpy as np

ALL_FIELDS = ["judgment_delegation", "cognitive_passivity", "information_dependency", "da_derived"]


def compute_slope(seq: list[float]) -> float:
    if len(seq) < 2:
        return 0.0
    x = np.arange(len(seq), dtype=float)
    x_norm = x / max(x.max(), 1)
    return float(np.polyfit(x_norm, seq, 1)[0])


def bucket3(v: float) -> int:
    return 0 if v <= 3 else (1 if v <= 6 else 2)


def score_trend(seq: list[float], threshold: float = 0.3) -> str:
    slope = compute_slope(seq)
    if slope > threshold:
        return "rising"
    elif slope < -threshold:
        return "falling"
    return "stable"


def bucket_trend(seq: list[float]) -> str:
    if len(seq) < 2:
        return "stable"
    mid = len(seq) // 2
    first = bucket3(float(np.mean(seq[:mid])))
    second = bucket3(float(np.mean(seq[mid:])))
    if second > first:
        return "rising"
    elif second < first:
        return "falling"
    return "stable"


def classify_trends(turn_scores: list[dict]) -> dict:
    score_trends = {}
    bucket_trends = {}
    slopes = {}

    for field in ALL_FIELDS:
        seq = [t["scores"][field] for t in turn_scores]
        slopes[field] = round(compute_slope(seq), 3)
        score_trends[field] = score_trend(seq)
        bucket_trends[field] = bucket_trend(seq)

    score_vote = Counter(score_trends.values()).most_common(1)[0][0]
    bucket_vote = Counter(bucket_trends.values()).most_common(1)[0][0]
    score_trends["overall"] = score_vote
    bucket_trends["overall"] = bucket_vote

    return {
        "score": score_trends,
        "bucket": bucket_trends,
        "slopes": slopes,
    }
