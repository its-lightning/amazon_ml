# Checkpointing (blocking stage)

`python -m src.build_candidates --split test` runs the blocking loop over every
Source 1 test entity (~1.7M) against the full Source 2/3 test files. At the rate this
pipeline achieves (roughly 70-90 entities/sec on this machine), that's realistically
**2.5-4+ hours**. This document explains how the run survives being interrupted —
Ctrl+C, a crash, a segfault, closing the terminal — without losing that time.

## Why this exists

Blocking on the full test set was tried without checkpointing and killed manually
after 24 minutes with no visible progress (the setup phase — fitting two TF-IDF
matrices and building an inverted index over ~10M rows — printed nothing for a long
time, which looked like a hang but wasn't). Restarting from zero after losing hours of
progress to any interruption is exactly what this avoids.

## How it works

While `generate_candidate_pairs` (`src/blocking.py`) runs, it:

1. Accumulates newly-generated candidate pairs in memory.
2. Every `--checkpoint-seconds` (default 300 = 5 min), flushes whatever has
   accumulated to a new small parquet file — `candidates/<split>_checkpoint/chunk_NNNNN.parquet`
   — and updates `candidates/<split>_checkpoint/state.json` with
   `{"next_index": <how many Source 1 entities are done>, "chunks": [...]}`. The
   in-memory buffer is cleared after each flush, so peak memory never holds more than
   one interval's worth of rows — not the whole multi-hour run's output.
3. On a clean finish, all chunks are merged (streamed, not loaded into memory all at
   once — see `_merge_chunks`) into the final `candidates/<split>_candidate_pairs.parquet`,
   and the checkpoint directory is deleted.
4. On `KeyboardInterrupt` (Ctrl+C) or any other Python-level exception, it flushes
   whatever's in the buffer *before* re-raising, so you lose at most the time since the
   last periodic flush — not the whole run. (A hard OS-level crash — a segfault — can't
   run this handler; in that case you lose only the time since the last periodic flush,
   up to `--checkpoint-seconds`.)

## How to use it

**It's automatic.** Just run the command:

```bash
python -m src.build_candidates --split test
```

If it gets interrupted for any reason, run the **exact same command again**. It
detects `candidates/test_checkpoint/` on startup, logs
`resuming from checkpoint: N/1,732,544 entities already done`, and continues from
entity N instead of entity 0.

Note: the setup phase (TF-IDF fit + index build, ~15-20 min at full test scale) is
**not** checkpointed and reruns on every resume — only the main per-entity loop skips
ahead. This is a deliberate simplification (checkpointing the fitted TF-IDF matrices
and index dicts would mean serializing multi-GB scipy/dict structures); redoing ~20
minutes of setup is a much smaller tax than redoing hours of loop.

To start over from scratch and discard an existing checkpoint (e.g. after changing
`--top-k`):

```bash
python -m src.build_candidates --split test --fresh
```

To change how often it flushes (shorter = less work lost per interruption, more I/O
overhead):

```bash
python -m src.build_candidates --split test --checkpoint-seconds 120
```

## Checking progress without waiting for a log line

The chunk files are plain parquet and readable at any time while the run is still
going:

```python
import pandas as pd, json
state = json.load(open("candidates/test_checkpoint/state.json"))
print(state["next_index"], "/", 1_732_544, "entities done")
print(len(state["chunks"]), "chunks flushed so far")
```

Or just watch the terminal — every phase now logs a timestamped line (source loading,
TF-IDF fitting, index building progress every ~10%, and the `blocking: X%|...|ETA` tqdm
bar once the main loop starts), so a silent terminal for more than a few minutes is a
real signal something's wrong, not just an artifact of no progress output existing.

## `build_features.py` and `predict.py`

These are not checkpointed — they're much faster (feature computation on ~6-12M pairs
took under 2 minutes in testing; the classifier's scoring pass is faster still) and
were not the source of the multi-hour risk. They do have the same timestamped logging
as `build_candidates.py` so a stall is visible either way.
