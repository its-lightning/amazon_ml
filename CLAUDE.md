# Amazon ML Challenge 2026 — Business Entity Resolution

## Project overview

Given noisy business records from three independent sources (Source 1, 2, 3), find
every Source 2/Source 3 record that refers to the same real-world business as each
Source 1 record. Source 1 is the deduplicated reference set: each Source 1 entity may
match zero, one, or many records in Source 2/3. The pipeline has two stages:

1. **Blocking / candidate generation** — cut the O(n×m) comparison space down to a
   small candidate set per Source 1 entity. This is graded on its own (candidate set
   size and recall ceiling), not just via the final leaderboard score.
2. **Matching** — score/filter each candidate pair down to final matches.

Scoring is **F_0.5** (precision weighted 2x over recall), so a false merge is punished
harder than a missed match — bias thresholds toward precision, and don't guess when
unsure.

## Current status (updated 2026-09-26 — read this first on a fresh session)

**Pipeline: built and working.** `code/business_entity_resolution/src/` has the full
blocking → features → train → predict flow (see that folder's `README.md` for exact
commands). Verified end-to-end multiple times on sampled data.

**Matcher: already trained.** A model exists at
`code/business_entity_resolution/models/matcher.json` (+ `feature_columns.json`,
`decision_threshold.json`) trained on a 150K-entity bounded sample
(`src/make_training_sample.py` — see "Environment / setup" below for why sampling was
necessary). Validation macro F_0.5 ≈ **0.835**, decision threshold ≈ 0.6. These
`models/` files are **gitignored — local to this machine only**, not in git. If you're
on a different machine/clone, this model doesn't exist yet and needs retraining (open
`notebooks/02_train_matcher.ipynb`, run all cells — see that notebook's prerequisites).

**Not yet done: test-set inference.** `output/matching_results.tsv` and
`output/candidate_pairs.tsv` do not exist yet. This is the next real task. It requires
`python -m src.build_candidates --split test` → `build_features.py --split test` →
`predict.py`, run from `code/business_entity_resolution/`. **This step alone takes
2.5-4+ hours** (full test set: 1.73M/4.9M/5.1M rows across source1/2/3, no sampling
allowed here — every test Source 1 entity needs a prediction). Two things to know
before running it:

1. **Checkpointed and resumable** — see `code/business_entity_resolution/CHECKPOINTING.md`.
   If interrupted (Ctrl+C, crash, closing the terminal), just re-run the exact same
   command; it detects the checkpoint and resumes instead of restarting. `--fresh`
   discards a checkpoint to start clean.
2. **Check free RAM first.** The one real attempt at this run so far was killed by
   Claude Code's own memory-pressure protection — not a bug in the pipeline — because
   the *system* had only ~5GB free out of 16GB total (the rest was other running
   programs: a game, Discord, browser tabs, IDE — nothing to do with this pipeline).
   Before starting, check `Get-Process | Sort-Object WorkingSet64 -Descending | Select
   -First 10 Name, @{N='MemGB';E={[math]::Round($_.WorkingSet64/1GB,2)}}` in PowerShell
   and close whatever's eating RAM if headroom looks tight. Also pass `--top-k 10`
   (default is 20) — the default produces an estimated ~135M candidate pairs on the
   full test set (extrapolated from the training sample's ratio), which is both worse
   for the "smaller candidate set scores higher" criterion and a real memory risk
   downstream in `build_features.py` (not checkpointed, not chunked — holds the whole
   feature table in memory). `--top-k 10` roughly halves that.

**Known gotchas already fixed** (don't reintroduce these):
- `TfidfVectorizer` OOMs on the full corpus (builds one global vocab dict before
  applying `max_features` — fine at 1.4M strings, blows up at 11.7M). Fixed: use
  `HashingVectorizer` + `TfidfTransformer` instead (`src/blocking.py`).
- The original `downsample_negatives` (per-entity Python loop appending ~300K tiny
  frames) crashed outright at scale. Fixed: fully vectorized (shuffle + `cumcount` +
  one concat) in `src/train_matcher.py`.
- Two segfaults (no Python traceback) occurred generating candidates at ~150K-entity
  scale before pool-size caps (`MAX_CANDIDATE_POOL_SIZE` in `config.py`) and
  `float32` dtypes were added — if you see a bare segfault again, suspect unbounded
  sparse-matrix operations or per-entity candidate pools, not a logic bug.

**Next steps in order:** free RAM → run the 3-step test inference above (checkpointed,
so safe to interrupt) → validate (`utils/validate_submission.py`) → upload
`matching_results.tsv` to the leaderboard portal (only the user can do this) → fill in
`Documentation_template.md` → assemble the final submission zip. See "Ideas for
improvement" in `code/business_entity_resolution/README.md` for score-improvement
options once a first full submission exists.

## Data

Location (do not move — referenced in place, not copied into this repo):

```
6ab10eb3b23ba_student_resource/student_resource/
├── README.md                     official problem statement (verbatim source of truth)
├── Documentation_template.md     methodology write-up template — fill in for final submission
├── utils/validate_submission.py  stdlib-only output validator, run before treating output as done
└── dataset/
    ├── train/
    │   ├── train_source1.tsv         Source 1 training records
    │   ├── train_source2.tsv         Source 2 training records
    │   ├── train_source3.tsv         Source 3 training records
    │   └── train_ground_truth.tsv    source1_entity_id -> matched_entity_ids (ground truth)
    └── test/
        ├── test_source1.tsv          Source 1 test records — every one needs a prediction
        ├── test_source2.tsv
        └── test_source3.tsv
```

This directory is **gitignored** (~2.3GB, 26M+ rows across 7 files) — never try to add
it to git. Same for `6ab10eb3b23ba_student_resource.zip` at the repo root (the original
download).

**Schema** (`*_source1/2/3.tsv`, tab-separated):

| column | notes |
|---|---|
| `entity_id` | prefix `S1-`/`S2-`/`S3-` indicates source; there is no separate source column |
| `business_name` | abbreviations, legal-suffix inconsistencies, DBA names, typos, word-order transpositions, punctuation (`&` vs `and`) |
| `business_address` | partial addresses, abbreviations (`Rd`/`Road`), transliteration variants, missing PIN/state, landmark references, reordered components |
| `country` | **open-set string label** — train has only `US`/`India`; **test additionally has `France`**. Never hardcode, filter, or one-hot to `{US, India}` — every test entity, France included, must get a prediction |

**Critical gotcha:** these are `.tsv`, not `.csv` — addresses and ID lists contain
commas. Always read with `sep="\t"` explicitly:
```python
df = pd.read_csv("path/to/file.tsv", sep="\t")
```
Reading without it silently collapses everything into one column — no error, just
wrong data.

`train_ground_truth.tsv` columns: `source1_entity_id`, `matched_entity_ids`
(comma-separated `entity_id`s from Source 2/3, empty when no match).

**No test ground truth is provided.** Hold out your own validation split from
`train/` and score it yourself with the F_0.5 formula below before relying on the
public leaderboard.

## Required outputs

Both files go in `output/` (gitignored — regenerate, don't commit the TSVs
themselves; the directory structure is tracked via `.gitkeep`):

### `output/matching_results.tsv` — the only file scored on the leaderboard
| column | description |
|---|---|
| `source1_entity_id` | a Source 1 `entity_id` |
| `matched_entity_ids` | comma-separated Source 2/3 `entity_id`s, empty if no match |

### `output/candidate_pairs.tsv` — your blocking stage's final candidate set
| column | description |
|---|---|
| `source1_entity_id` | a Source 1 `entity_id` |
| `candidate_entity_ids` | comma-separated Source 2/3 `entity_id`s the matcher actually scored |

This must be the **last** filtering stage before the matching model runs — not an
earlier, wider blocking pass. Every ID in `matching_results.tsv` must also appear in
the corresponding row of `candidate_pairs.tsv` (a match absent from candidates signals
a pipeline bug).

**Format rules for both files** (violations get the submission rejected, not just
scored low):
- Exactly one row per Source 1 entity in the test set — every entity, no exceptions.
- Empty string (not `"none"`, not `"[]"`) when there's no match/candidate.
- Comma-separated, no quoting.
- No duplicate IDs within a single row's list.
- No duplicate `source1_entity_id` rows.
- Only `S2-`/`S3-` IDs that exist in the test set — no self-matches to `S1-`, no
  invented IDs.

**Always validate before calling output done:**
```bash
python3 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
It prints `PASS` (exit 0) or a numbered issue list (exit 1). It does not compute your
score — only checks format/structural validity.

## Scoring

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
Computed **per Source 1 entity, then macro-averaged**. A true singleton predicted as
empty scores 1.0; predicting any match for a true singleton scores 0.0. Precision is
weighted 2x over recall — when in doubt, drop a marginal candidate rather than keep it.

## Hard constraints (disqualifying, not just score-reducing)

- **No external data/API lookups of any kind** — no commercial ER services, no
  government business-registry lookups, no geocoding APIs, no internet-sourced
  augmentation. Training data only. Any evidence of this causes immediate
  disqualification, and all code/methodology gets reviewed for it.
- Final model must be **MIT or Apache-2.0 licensed** and **≤8B parameters**.
- Every test Source 1 entity must appear in the output exactly once.

## Submission package layout

The final deliverable is a zip; this repo's structure mirrors it directly so the zip
can be assembled straight from here:

```
code/business_entity_resolution/
├── src/                 all pipeline source code
├── README.md            exact reproduce-from-scratch instructions (data → blocking → matching → output)
└── requirements.txt     pinned dependencies
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```
Plus the filled-in `Documentation_template.md` (copy from
`6ab10eb3b23ba_student_resource/student_resource/Documentation_template.md`, fill in
methodology/blocking strategy/model architecture/results — no page limit, prioritize
depth) — keep this at the repo root and copy it into the zip at submission time.

## Git workflow (required — follow without being asked)

- **Commit after every meaningful change**: a new script, a pipeline stage, a bugfix,
  a validated experiment. Don't batch unrelated changes into one commit.
- **Push to `origin` after every commit** — don't leave local commits unpushed.
- **Never commit**: the raw dataset, the original `.zip`, model checkpoints, or
  generated `output/*.tsv` files. `.gitignore` already excludes these — if a new
  category of large/generated file shows up, add it to `.gitignore` rather than
  committing it.
- Descriptive commit messages (what changed and why), one logical change per commit.
- Never force-push; never rewrite already-pushed history.
- If `git push` fails because there's no remote configured yet, say so rather than
  silently skipping the push — the remote may need to be (re)created.

## Environment / setup

- Python 3.10+ recommended (validator itself only needs 3.8+, stdlib only).
- Install pipeline dependencies: `pip install -r code/business_entity_resolution/requirements.txt`
  (populate this file as dependencies are added — pin versions).
- End-to-end run instructions live in `code/business_entity_resolution/README.md` —
  keep it in sync with the actual entry point(s) as the pipeline is built.
- Full training data is ~2.3GB / 12.5M rows across the three train sources; the current
  blocking implementation is a per-entity Python loop, so on a memory-constrained
  machine (this dev machine: 24-core CPU, RTX 4060 8GB, only ~16GB RAM) it's
  impractical to run over the full train set directly. `src/make_training_sample.py`
  builds a bounded, representative sample instead (real positives from ground truth +
  a random negative/background pool) — see `code/business_entity_resolution/README.md`
  for the exact commands. **This sampling is for training the matcher only** — final
  test-set inference must still cover every test Source 1 entity, never sampled.
- XGBoost training uses the GPU by default (`device="cuda"` in `train_xgboost()`,
  `code/business_entity_resolution/src/train_matcher.py`); pass `use_gpu=False` on a
  machine without a CUDA GPU.
- Validator command: see "Required outputs" above.

## Fair play reminder

The external-lookup ban is enforced by code/methodology review, not just automated
checks — never add geocoding, business-registry, or other internet-sourced lookups to
the pipeline, even for "just normalizing an address." Everything must come from the
provided training data.
