from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(ece)


def worst_group_accuracy(
    predictions: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> float:
    values = []
    for group in np.unique(groups):
        mask = groups == group
        if mask.any():
            values.append(float((predictions[mask] == labels[mask]).mean()))
    return min(values) if values else float("nan")


def conditional_equal_opportunity_gap(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    need_bins: np.ndarray,
) -> float:
    true_prob = probabilities[np.arange(labels.shape[0]), labels]
    gaps = []
    for need in np.unique(need_bins):
        means = []
        need_mask = need_bins == need
        for group in np.unique(groups[need_mask]):
            mask = need_mask & (groups == group)
            if mask.sum() >= 2:
                means.append(float(true_prob[mask].mean()))
        if len(means) >= 2:
            gaps.append(max(means) - min(means))
    return float(np.mean(gaps)) if gaps else 0.0


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    need_bins: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "worst_group_accuracy": worst_group_accuracy(predictions, labels, groups),
        "ece": expected_calibration_error(probabilities, labels),
        "conditional_eo_gap": conditional_equal_opportunity_gap(
            probabilities, labels, groups, need_bins
        ),
    }


def risk_coverage_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    coverages: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> list[dict[str, float]]:
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(scores)
    n = labels.shape[0]
    result = []
    for coverage in coverages:
        accepted = order[: max(1, int(round(n * coverage)))]
        risk = 1.0 - float((predictions[accepted] == labels[accepted]).mean())
        result.append({"coverage": float(coverage), "risk": risk})
    return result
