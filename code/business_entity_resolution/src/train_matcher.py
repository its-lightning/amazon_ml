"""Training helpers for the candidate-pair matcher (XGBoost, Apache-2.0, far under the
8B-parameter cap).

These functions are called from ``notebooks/02_train_matcher.ipynb`` — this module
intentionally has no ``if __name__ == "__main__"`` training entry point, since the
training run itself is done interactively in the notebook.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import (
    FEATURE_LIST_PATH,
    MATCHER_MODEL_PATH,
    NEGATIVE_SAMPLES_PER_POSITIVE,
    RANDOM_SEED,
    THRESHOLD_PATH,
    VALIDATION_FRACTION,
)
from .features import FEATURE_COLUMNS


def label_candidates(features_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a binary ``label`` column: 1 if the candidate is a true match."""
    truth_pairs = {
        (row.source1_entity_id, cid)
        for row in ground_truth_df.itertuples(index=False)
        for cid in row.matched_entity_ids
    }
    pair_keys = list(zip(features_df["source1_entity_id"], features_df["candidate_entity_id"]))
    labels = np.fromiter((1 if k in truth_pairs else 0 for k in pair_keys), dtype=np.int8, count=len(pair_keys))
    out = features_df.copy()
    out["label"] = labels
    return out


def split_entities_by_source1(s1_ids, validation_fraction: float = VALIDATION_FRACTION, seed: int = RANDOM_SEED):
    """Split Source 1 entity ids into train/validation sets (entity-level split, so
    all candidate pairs for one entity stay on the same side)."""
    ids = np.array(sorted(set(s1_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * (1 - validation_fraction))
    return set(ids[:cut]), set(ids[cut:])


def downsample_negatives(
    labeled_df: pd.DataFrame,
    negatives_per_positive: int = NEGATIVE_SAMPLES_PER_POSITIVE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Cap the number of negative (non-match) candidates kept per Source 1 entity,
    relative to how many positives it has (at least 1, so pure-negative/singleton
    entities still contribute some negative examples). Speeds up training without
    throwing away the (usually far more informative) positive pairs.

    Vectorized (shuffle once + groupby.cumcount + a single concat) rather than a
    per-entity Python loop appending hundreds of thousands of tiny frames — the
    latter is a known memory/perf anti-pattern at this scale (millions of rows,
    100k+ groups) and crashed outright on this machine's 16GB RAM.
    """
    is_pos = labeled_df["label"].to_numpy() == 1
    pos_df = labeled_df.loc[is_pos]
    neg_df = labeled_df.loc[~is_pos].sample(frac=1.0, random_state=seed)

    pos_counts = pos_df.groupby("source1_entity_id").size()
    limits = neg_df["source1_entity_id"].map(pos_counts).fillna(0).astype(int) * negatives_per_positive
    limits = limits.clip(lower=negatives_per_positive)

    rank = neg_df.groupby("source1_entity_id").cumcount()
    kept_neg_df = neg_df[rank.to_numpy() < limits.to_numpy()]

    return pd.concat([pos_df, kept_neg_df], ignore_index=True)


def train_xgboost(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    use_gpu: bool = True,
    early_stopping_rounds: int | None = 30,
    **xgb_params,
) -> xgb.XGBClassifier:
    """Fit the matcher. ``train_df`` (and ``eval_df`` if given) must have
    FEATURE_COLUMNS + a ``label`` column. Pass ``eval_df`` to get per-round
    validation metrics (and early stopping) so you can see overfitting as it happens
    instead of only after the fit finishes."""
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    if use_gpu:
        params.update(device="cuda", tree_method="hist")
    if eval_df is not None and early_stopping_rounds:
        params["early_stopping_rounds"] = early_stopping_rounds
    params.update(xgb_params)

    model = xgb.XGBClassifier(**params)
    fit_kwargs = {}
    if eval_df is not None:
        fit_kwargs["eval_set"] = [
            (train_df[FEATURE_COLUMNS], train_df["label"]),
            (eval_df[FEATURE_COLUMNS], eval_df["label"]),
        ]
        fit_kwargs["verbose"] = True
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"], **fit_kwargs)
    return model


def score_candidates(model: xgb.XGBClassifier, features_df: pd.DataFrame) -> pd.DataFrame:
    """Return ``features_df`` plus a ``score`` column (P(match))."""
    out = features_df.copy()
    out["score"] = model.predict_proba(out[FEATURE_COLUMNS])[:, 1]
    return out


def save_model(model: xgb.XGBClassifier, threshold: float) -> None:
    MATCHER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MATCHER_MODEL_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))


def load_model() -> tuple[xgb.XGBClassifier, float]:
    model = xgb.XGBClassifier()
    model.load_model(MATCHER_MODEL_PATH)
    threshold = json.loads(THRESHOLD_PATH.read_text())["threshold"]
    return model, threshold
