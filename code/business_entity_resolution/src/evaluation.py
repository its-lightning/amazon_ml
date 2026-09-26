"""F_0.5 macro scoring (matches the official evaluation) and threshold selection.

Used by the training notebook to pick a decision threshold that is biased toward
precision (as F_0.5 rewards), and by anyone who wants to score their own held-out
validation split the same way the leaderboard will.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

BETA_SQUARED = 0.25  # beta = 0.5 -> beta^2 = 0.25


def f0_5_per_entity(predicted_ids: set, true_ids: set) -> float:
    """F_0.5 for one Source 1 entity. Empty-true/empty-pred (correct singleton) = 1.0."""
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0
    if not predicted_ids:
        return 0.0
    tp = len(predicted_ids & true_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    return (1 + BETA_SQUARED) * precision * recall / (BETA_SQUARED * precision + recall)


def macro_f0_5(predictions: dict, truth: dict) -> float:
    """Macro-average F_0.5 across every key in ``truth`` (missing predictions -> empty)."""
    if not truth:
        return 0.0
    scores = [f0_5_per_entity(set(predictions.get(s1, set())), set(true_ids)) for s1, true_ids in truth.items()]
    return float(np.mean(scores))


def predictions_from_scores(scored_df: pd.DataFrame, threshold: float) -> dict:
    """Build {source1_entity_id: set(candidate_entity_id)} from scores >= threshold."""
    matched = scored_df[scored_df["score"] >= threshold]
    preds = defaultdict(set)
    for s1_id, group in matched.groupby("source1_entity_id")["candidate_entity_id"]:
        preds[s1_id] = set(group)
    return dict(preds)


def find_best_threshold(scored_df: pd.DataFrame, truth: dict, thresholds=None) -> tuple[float, float]:
    """Grid-search the score threshold that maximizes macro F_0.5 on ``truth``.

    ``scored_df`` needs columns source1_entity_id, candidate_entity_id, score.
    ``truth`` is {source1_entity_id: iterable_of_true_candidate_ids} and should cover
    every Source 1 entity in the validation split, including true singletons (empty
    iterable) so they're scored too.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.025)

    truth = {s1: set(ids) for s1, ids in truth.items()}
    best_threshold, best_score = 0.5, -1.0
    for t in thresholds:
        preds = predictions_from_scores(scored_df, t)
        score = macro_f0_5(preds, truth)
        if score > best_score:
            best_score, best_threshold = score, float(t)
    return best_threshold, best_score
