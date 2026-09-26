"""CLI: compute pairwise similarity features for a split's candidate pairs.

Usage (from code/business_entity_resolution/, after build_candidates.py):
    python -m src.build_features --split train
    python -m src.build_features --split test

For --split train, also joins train_ground_truth.tsv to attach a ``label`` column and
(by default) downsamples easy negatives per Source 1 entity for faster training.
"""

import argparse

import pandas as pd

from . import config
from .blocking import prepare_records
from .features import compute_pair_features
from .io_utils import read_ground_truth, read_source
from .train_matcher import downsample_negatives, label_candidates

SPLIT_SOURCE1 = {"train": config.TRAIN_SOURCE1, "test": config.TEST_SOURCE1}
SPLIT_SOURCE2 = {"train": config.TRAIN_SOURCE2, "test": config.TEST_SOURCE2}
SPLIT_SOURCE3 = {"train": config.TRAIN_SOURCE3, "test": config.TEST_SOURCE3}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute candidate-pair similarity features.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--no-downsample", action="store_true", help="Keep all negatives (train split only).")
    args = parser.parse_args()

    candidates_path = config.CANDIDATES_DIR / f"{args.split}_candidate_pairs.parquet"
    if not candidates_path.exists():
        raise SystemExit(f"{candidates_path} not found — run build_candidates.py --split {args.split} first.")
    pairs_df = pd.read_parquet(candidates_path)

    print(f"Loading {args.split} sources...")
    s1_df = prepare_records(read_source(SPLIT_SOURCE1[args.split]))
    s2_df = prepare_records(read_source(SPLIT_SOURCE2[args.split]))
    s3_df = prepare_records(read_source(SPLIT_SOURCE3[args.split]))

    print(f"Computing features for {len(pairs_df):,} candidate pairs...")
    features_df = compute_pair_features(pairs_df, s1_df, {"S2": s2_df, "S3": s3_df})

    if args.split == "train":
        ground_truth = read_ground_truth(config.TRAIN_GROUND_TRUTH)
        features_df = label_candidates(features_df, ground_truth)
        print(f"  positives: {features_df['label'].sum():,} / {len(features_df):,}")
        if not args.no_downsample:
            before = len(features_df)
            features_df = downsample_negatives(features_df)
            print(f"  downsampled negatives: {before:,} -> {len(features_df):,} rows")

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FEATURES_DIR / f"{args.split}_features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
