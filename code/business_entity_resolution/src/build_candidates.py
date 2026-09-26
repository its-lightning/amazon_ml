"""CLI: run the blocking stage and write candidate pairs.

Usage (from code/business_entity_resolution/):
    python -m src.build_candidates --split train
    python -m src.build_candidates --split train --source sample  # after make_training_sample.py
    python -m src.build_candidates --split test

Writes a long-format parquet to candidates/<split>_candidate_pairs.parquet for reuse
by build_features.py / predict.py. For --split test it additionally writes the
required output/candidate_pairs.tsv (wide format, one row per Source 1 test entity).
"""

import argparse

from . import config
from .blocking import candidates_to_wide, generate_candidate_pairs, prepare_records
from .io_utils import read_source, write_id_list_tsv

SPLIT_PATHS = {
    "train": (config.TRAIN_SOURCE1, config.TRAIN_SOURCE2, config.TRAIN_SOURCE3),
    "sample": (config.SAMPLE_TRAIN_SOURCE1, config.SAMPLE_TRAIN_SOURCE2, config.SAMPLE_TRAIN_SOURCE3),
    "test": (config.TEST_SOURCE1, config.TEST_SOURCE2, config.TEST_SOURCE3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blocking candidate pairs.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--source", choices=["full", "sample"], default="full",
        help="For --split train: 'sample' uses the bounded set from make_training_sample.py "
        "(recommended on memory-constrained machines; run that script first).",
    )
    parser.add_argument("--top-k", type=int, default=config.TOP_K_PER_SOURCE)
    parser.add_argument("--min-similarity", type=float, default=config.MIN_NAME_SIMILARITY)
    args = parser.parse_args()

    split_key = "sample" if (args.split == "train" and args.source == "sample") else args.split
    s1_path, s2_path, s3_path = SPLIT_PATHS[split_key]
    print(f"Loading {split_key} sources...")
    s1_df = prepare_records(read_source(s1_path))
    s2_df = prepare_records(read_source(s2_path))
    s3_df = prepare_records(read_source(s3_path))
    print(f"  Source 1: {len(s1_df):,} rows | Source 2: {len(s2_df):,} rows | Source 3: {len(s3_df):,} rows")

    pairs_df = generate_candidate_pairs(
        s1_df,
        {"S2": s2_df, "S3": s3_df},
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    print(f"Generated {len(pairs_df):,} candidate pairs for {pairs_df['source1_entity_id'].nunique():,} "
          f"Source 1 entities (of {len(s1_df):,} total).")

    config.CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CANDIDATES_DIR / f"{split_key}_candidate_pairs.parquet"
    pairs_df.to_parquet(out_path, index=False)
    print(f"Wrote long-format candidates to {out_path}")

    if args.split == "test":
        wide = candidates_to_wide(pairs_df, s1_df["entity_id"])
        write_id_list_tsv(config.CANDIDATE_PAIRS_PATH, wide, "source1_entity_id", "candidate_entity_ids")
        print(f"Wrote {config.CANDIDATE_PAIRS_PATH}")


if __name__ == "__main__":
    main()
