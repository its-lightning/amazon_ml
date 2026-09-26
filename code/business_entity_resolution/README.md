# business_entity_resolution

Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. See the
repo root `CLAUDE.md` for the full problem spec, data schema, and output format.

## Approach

- **Blocking (candidate generation):** partition by exact `country` match, then an
  inverted index over normalized business-name tokens, metaphone phonetic codes,
  address tokens, and postal codes finds a bounded candidate pool per Source 1 entity
  (`src/blocking.py`). That pool is re-ranked in two independent TF-IDF cosine spaces —
  name and address — and the top-K by name is unioned with the top-K by address
  (not one shared ranking by max similarity: that let "same address, different
  business" confusers crowd true name-matches out of the cutoff in testing).
- **Matching:** hand-engineered similarity features (TF-IDF cosine, Levenshtein ratio,
  token Jaccard, phonetic overlap, postal-code match, ...) computed per candidate pair
  (`src/features.py`), scored by an XGBoost classifier (Apache-2.0, far under the 8B
  parameter cap) trained on the labeled training candidates (`src/train_matcher.py`).
  The decision threshold is chosen by grid search to maximize macro F_0.5 on a held-out
  validation split of Source 1 entities (`src/evaluation.py`).
- **Checkpoint (2026-09-26):** on the 150K-entity training sample (see below), the
  matcher reaches validation macro F_0.5 ≈ 0.835 with roughly balanced feature
  importance between name (~50%) and address (~30%) similarity. This is a baseline,
  not a final number — see "Ideas for improvement" below.

## Setup

```bash
pip install -r requirements.txt
```

## Run end-to-end

All commands run from this directory (`code/business_entity_resolution/`).

```bash
# 0. (Recommended on <32GB RAM machines) Build a bounded training sample instead of
#    loading the full 2.2M/5M/5.3M-row train files through the current per-entity
#    Python blocking loop. Real positives (from ground truth) are always included, plus
#    a random background pool for realistic negatives. Tune sizes to your hardware.
python -m src.make_training_sample --n-positive 120000 --n-singleton 30000 --n-background 400000

# 1. Blocking — writes candidates/{split}_candidate_pairs.parquet, and for
#    --split test also writes ../../output/candidate_pairs.tsv
python -m src.build_candidates --split train --source sample   # or --source full
python -m src.build_candidates --split test                    # always the full test set

# 2. Feature engineering — writes features/{split}_features.parquet
#    (train split also joins the matching ground-truth file to attach labels)
python -m src.build_features --split train --source sample     # or --source full
python -m src.build_features --split test

# 3. Train the matcher and pick a decision threshold — run interactively:
#    notebooks/02_train_matcher.ipynb (set USE_SAMPLE = True/False near the top to match step 1-2)
#    Uses the GPU by default (train_xgboost(..., use_gpu=True)); pass use_gpu=False if you have no CUDA GPU.
#    Saves models/matcher.json, models/feature_columns.json, models/decision_threshold.json

# 4. Inference — writes ../../output/matching_results.tsv
python -m src.predict

# 5. Validate before submitting
python "../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py" \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir "../../6ab10eb3b23ba_student_resource/student_resource/dataset/test"
```

`notebooks/01_eda.ipynb` is optional exploratory analysis, not part of the reproduce
path. Step 4 (inference) always uses the full test set regardless of whether you
trained on the sample or the full train data — sampling only ever applies to training.

## Ideas for improvement

At the 2026-09-26 checkpoint, ~50% of true matches for positive entities are still
missed (false negatives dominate over false positives, which is the right side to err
on for F_0.5, but there's real headroom):

- **Run on the full training data** instead of the 150K-entity sample — more positives,
  more realistic negative distribution. Needs either more RAM/time or a faster
  (vectorized/batched, not per-entity Python loop) blocking implementation.
- **Increase `TOP_K_PER_SOURCE`** (currently 20) or `MAX_CANDIDATE_POOL_SIZE` (currently
  2000) — cheap to try, directly raises the blocking recall ceiling at the cost of a
  larger (worse-ranked) candidate set.
- **Parse address components** (city/state/PIN) explicitly instead of relying on token
  Jaccard over the whole cleaned address string — a structured city/state match feature
  would likely be more robust than raw token overlap.
- **Tune `NEGATIVE_SAMPLES_PER_POSITIVE`** (currently 5) and try `scale_pos_weight` in
  `train_xgboost` — the current class balance was not tuned, just a reasonable default.
- **Try LightGBM** as a second model and compare — same license/size profile as XGBoost,
  sometimes faster to train with categorical-like features.

## Layout

```
src/
├── config.py          paths + hyperparameters (all paths resolved relative to this file)
├── io_utils.py         tsv read/write helpers (explicit sep="\t", no NaN-sniffing)
├── normalize.py        name/address normalization (transliteration, legal suffixes, abbreviations)
├── blocking.py         candidate generation (inverted index + TF-IDF re-rank)
├── features.py         pairwise similarity feature engineering
├── train_matcher.py    labeling, train/val split, XGBoost train/save/load helpers (GPU by default)
├── evaluation.py        F_0.5 macro scorer + threshold search (matches official scoring)
├── make_training_sample.py  CLI: build a bounded, representative training sample
├── build_candidates.py  CLI: blocking stage
├── build_features.py    CLI: feature engineering stage
└── predict.py            CLI: inference -> matching_results.tsv
notebooks/
├── 01_eda.ipynb          exploratory data analysis
└── 02_train_matcher.ipynb  train the matcher, tune threshold, save the model
```

`models/`, `features/`, `candidates/`, and `data_sample/` under this directory are
gitignored (intermediate/large artifacts, regenerable from the commands above).
