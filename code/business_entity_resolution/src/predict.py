"""CLI: score test candidates with the trained matcher and write matching_results.tsv.

Usage (from code/business_entity_resolution/, after build_candidates.py --split test,
build_features.py --split test, and training the model in
notebooks/02_train_matcher.ipynb):
    python -m src.predict
"""

import pandas as pd

from . import config
from .evaluation import predictions_from_scores
from .io_utils import read_source, write_id_list_tsv
from .train_matcher import load_model, score_candidates


def main() -> None:
    features_path = config.FEATURES_DIR / "test_features.parquet"
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found — run build_features.py --split test first.")
    if not config.MATCHER_MODEL_PATH.exists():
        raise SystemExit(
            f"{config.MATCHER_MODEL_PATH} not found — train the matcher in "
            "notebooks/02_train_matcher.ipynb first (it saves the model there)."
        )

    features_df = pd.read_parquet(features_path)
    model, threshold = load_model()
    print(f"Loaded matcher, decision threshold={threshold}")

    scored = score_candidates(model, features_df)
    predictions = predictions_from_scores(scored, threshold)

    test_s1_ids = read_source(config.TEST_SOURCE1)["entity_id"]
    rows = {s1_id: predictions.get(s1_id, set()) for s1_id in test_s1_ids}

    write_id_list_tsv(config.MATCHING_RESULTS_PATH, rows, "source1_entity_id", "matched_entity_ids")
    n_matched = sum(1 for ids in rows.values() if ids)
    print(f"Wrote {config.MATCHING_RESULTS_PATH}: {n_matched:,} / {len(rows):,} entities with >=1 match")

    if not config.CANDIDATE_PAIRS_PATH.exists():
        print(
            f"WARNING: {config.CANDIDATE_PAIRS_PATH} does not exist yet — run "
            "build_candidates.py --split test to produce it before submitting."
        )


if __name__ == "__main__":
    main()
