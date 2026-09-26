# business_entity_resolution

Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. See the
repo root `CLAUDE.md` for the full problem spec, data schema, and output format.

## Approach

- **Blocking (candidate generation):** partition by exact `country` match, then an
  inverted index over normalized business-name tokens + metaphone phonetic codes finds
  a bounded candidate pool per Source 1 entity; that pool is re-ranked with character
  n-gram TF-IDF cosine similarity and truncated to the top-K per source
  (`src/blocking.py`).
- **Matching:** hand-engineered similarity features (TF-IDF cosine, Levenshtein ratio,
  token Jaccard, phonetic overlap, postal-code match, ...) computed per candidate pair
  (`src/features.py`), scored by an XGBoost classifier (Apache-2.0, far under the 8B
  parameter cap) trained on the labeled training candidates (`src/train_matcher.py`).
  The decision threshold is chosen by grid search to maximize macro F_0.5 on a held-out
  validation split of Source 1 entities (`src/evaluation.py`).

## Setup

```bash
pip install -r requirements.txt
```

## Run end-to-end

All commands run from this directory (`code/business_entity_resolution/`).

```bash
# 1. Blocking — writes candidates/{split}_candidate_pairs.parquet, and for
#    --split test also writes ../../output/candidate_pairs.tsv
python -m src.build_candidates --split train
python -m src.build_candidates --split test

# 2. Feature engineering — writes features/{split}_features.parquet
#    (train split also joins train_ground_truth.tsv to attach labels)
python -m src.build_features --split train
python -m src.build_features --split test

# 3. Train the matcher and pick a decision threshold — run interactively:
#    notebooks/02_train_matcher.ipynb
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
path.

## Layout

```
src/
├── config.py          paths + hyperparameters (all paths resolved relative to this file)
├── io_utils.py         tsv read/write helpers (explicit sep="\t", no NaN-sniffing)
├── normalize.py        name/address normalization (transliteration, legal suffixes, abbreviations)
├── blocking.py         candidate generation (inverted index + TF-IDF re-rank)
├── features.py         pairwise similarity feature engineering
├── train_matcher.py    labeling, train/val split, XGBoost train/save/load helpers
├── evaluation.py        F_0.5 macro scorer + threshold search (matches official scoring)
├── build_candidates.py  CLI: blocking stage
├── build_features.py    CLI: feature engineering stage
└── predict.py            CLI: inference -> matching_results.tsv
notebooks/
├── 01_eda.ipynb          exploratory data analysis
└── 02_train_matcher.ipynb  train the matcher, tune threshold, save the model
```

`models/`, `features/`, and `candidates/` under this directory are gitignored
(intermediate/large artifacts, regenerable from the commands above).
