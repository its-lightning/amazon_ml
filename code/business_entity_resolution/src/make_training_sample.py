"""Build a bounded, representative training sample.

The full training data (2.2M Source 1 + 5M/5.3M Source 2/3 rows) is too large to
comfortably run through the current (pure-Python, per-entity) blocking loop on a
memory-constrained machine, and training the matcher does not need every row — a
large sample of real positives plus a big pool of background/negative records already
gives the classifier plenty of signal.

Sampling strategy (keeps recall honest, unlike pure random sampling of Source 2/3):
1. Take a random sample of Source 1 entities that DO have a ground-truth match
   (--n-positive) plus a random sample of true singletons (--n-singleton).
2. Pull in every Source 2/3 record that is a true match for those sampled entities
   (guarantees the positive pairs are actually present to be found by blocking).
3. Add a random background sample of Source 2/3 records (--n-background each) so
   blocking has realistic "hard negative" distractors to sift through, not just the
   true matches.

Usage (from code/business_entity_resolution/):
    python -m src.make_training_sample --n-positive 120000 --n-singleton 30000 --n-background 400000
"""

import argparse

import pandas as pd

from . import config
from .io_utils import read_ground_truth, read_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded training sample.")
    parser.add_argument("--n-positive", type=int, default=120_000, help="Source 1 entities WITH a match to sample.")
    parser.add_argument("--n-singleton", type=int, default=30_000, help="Source 1 entities WITHOUT a match to sample.")
    parser.add_argument("--n-background", type=int, default=400_000, help="Random background rows per source (2 and 3).")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    print("Loading train_ground_truth.tsv...")
    gt = read_ground_truth(config.TRAIN_GROUND_TRUTH)
    gt_pos = gt[gt["matched_entity_ids"].apply(len) > 0]
    gt_singleton = gt[gt["matched_entity_ids"].apply(len) == 0]

    gt_pos_sample = gt_pos.sample(n=min(args.n_positive, len(gt_pos)), random_state=args.seed)
    gt_singleton_sample = gt_singleton.sample(n=min(args.n_singleton, len(gt_singleton)), random_state=args.seed)
    gt_sample = pd.concat([gt_pos_sample, gt_singleton_sample], ignore_index=True)

    wanted_s1 = set(gt_sample["source1_entity_id"])
    wanted_other = {i for ids in gt_sample["matched_entity_ids"] for i in ids}
    print(f"sampled {len(wanted_s1):,} Source 1 entities "
          f"({len(gt_pos_sample):,} with matches, {len(gt_singleton_sample):,} singletons); "
          f"{len(wanted_other):,} guaranteed-positive Source 2/3 ids")

    print("Loading train_source1.tsv...")
    s1 = read_source(config.TRAIN_SOURCE1)
    s1_sample = s1[s1["entity_id"].isin(wanted_s1)]

    rng_state = args.seed
    for name, path, out_path in (
        ("source2", config.TRAIN_SOURCE2, config.SAMPLE_TRAIN_SOURCE2),
        ("source3", config.TRAIN_SOURCE3, config.SAMPLE_TRAIN_SOURCE3),
    ):
        print(f"Loading train_{name}.tsv...")
        df = read_source(path)
        must_keep = df[df["entity_id"].isin(wanted_other)]
        background = df[~df["entity_id"].isin(wanted_other)].sample(
            n=min(args.n_background, len(df)), random_state=rng_state
        )
        sampled = pd.concat([must_keep, background], ignore_index=True).drop_duplicates("entity_id")
        config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        sampled.to_csv(out_path, sep="\t", index=False)
        print(f"  wrote {len(sampled):,} rows ({len(must_keep):,} guaranteed matches + background) -> {out_path}")

    # Restrict ground-truth matched ids to whatever actually survived sampling.
    kept_other_ids = set()
    for path in (config.SAMPLE_TRAIN_SOURCE2, config.SAMPLE_TRAIN_SOURCE3):
        kept_other_ids |= set(read_source(path)["entity_id"])

    def _filter(ids):
        return [i for i in ids if i in kept_other_ids]

    gt_sample = gt_sample.copy()
    gt_sample["matched_entity_ids"] = gt_sample["matched_entity_ids"].apply(_filter)
    gt_sample["matched_entity_ids"] = gt_sample["matched_entity_ids"].apply(",".join)

    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    s1_sample.to_csv(config.SAMPLE_TRAIN_SOURCE1, sep="\t", index=False)
    gt_sample.to_csv(config.SAMPLE_TRAIN_GROUND_TRUTH, sep="\t", index=False)
    print(f"wrote {len(s1_sample):,} rows -> {config.SAMPLE_TRAIN_SOURCE1}")
    print(f"wrote {len(gt_sample):,} rows -> {config.SAMPLE_TRAIN_GROUND_TRUTH}")


if __name__ == "__main__":
    main()
