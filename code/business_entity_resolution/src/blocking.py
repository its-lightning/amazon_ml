"""Candidate generation (blocking).

Strategy: partition by exact `country` match (a business can't be in two countries),
then within each country build an inverted index from business-name tokens *and*
metaphone phonetic codes of those tokens (catches typos/spelling variants that don't
share exact tokens). For each Source 1 entity, union the index hits from its own
tokens/phonetic codes to get a bounded candidate pool, then re-rank that pool with
character n-gram TF-IDF cosine similarity and keep the top-K per source.

This keeps the search space bounded (no O(n*m) comparison) while giving the later
matching-model stage a high-recall candidate set to work with.
"""

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from tqdm import tqdm

from .config import (
    MAX_CANDIDATE_POOL_SIZE,
    MAX_TOKEN_DOC_FREQ,
    MIN_NAME_SIMILARITY,
    RANDOM_SEED,
    TOP_K_PER_SOURCE,
)
from .normalize import normalize_address, normalize_name


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized name/address columns used by blocking and feature engineering."""
    df = df.copy()
    name_norm = df["business_name"].apply(normalize_name)
    df["name_clean"] = name_norm.apply(lambda d: d["clean"])
    df["name_core"] = name_norm.apply(lambda d: d["core"])
    df["core_tokens"] = name_norm.apply(lambda d: d["core_tokens"])
    df["phonetic_codes"] = df["core_tokens"].apply(
        lambda toks: {jellyfish.metaphone(t) for t in toks if t}
    )

    addr_norm = df["business_address"].apply(normalize_address)
    df["address_clean"] = addr_norm.apply(lambda d: d["clean"])
    df["address_tokens"] = addr_norm.apply(lambda d: d["tokens"])
    df["postal_codes"] = addr_norm.apply(lambda d: d["postal_codes"])
    return df.reset_index(drop=True)


def _build_index(df: pd.DataFrame) -> dict:
    """Build a {(country, key): set(row_position)} inverted index.

    ``key`` is a name token, a ``"ph:<metaphone code>"`` phonetic key, an
    ``"addr:<token>"`` address token, or a ``"pin:<code>"`` postal code. Address
    tokens matter because feature importance shows address similarity is a stronger
    match signal than name similarity — blocking on name alone would silently drop
    true matches whose names were reworded/abbreviated but whose address matches.
    Keys that appear in more than ``MAX_TOKEN_DOC_FREQ`` of a country's rows are
    dropped (too common to be useful for blocking, e.g. leftover generic words).
    """
    index = defaultdict(set)
    country_counts = df["country"].value_counts().to_dict()

    n = len(df)
    report_every = max(1, n // 10)
    for pos, row in enumerate(df.itertuples(index=False)):
        if pos % report_every == 0:
            _log(f"    indexing row {pos:,}/{n:,} ({100 * pos // n}%)")
        country = row.country
        for tok in row.core_tokens:
            index[(country, tok)].add(pos)
        for code in row.phonetic_codes:
            index[(country, f"ph:{code}")].add(pos)
        for tok in row.address_tokens:
            index[(country, f"addr:{tok}")].add(pos)
        for pin in row.postal_codes:
            index[(country, f"pin:{pin}")].add(pos)

    max_per_country = {
        c: max(5, min(int(n * MAX_TOKEN_DOC_FREQ), MAX_CANDIDATE_POOL_SIZE))
        for c, n in country_counts.items()
    }
    for key in list(index):
        country, _ = key
        if len(index[key]) > max_per_country.get(country, 5):
            del index[key]
    return index


def _candidate_positions(row, index: dict) -> set:
    positions = set()
    country = row.country
    for tok in row.core_tokens:
        positions |= index.get((country, tok), set())
    for code in row.phonetic_codes:
        positions |= index.get((country, f"ph:{code}"), set())
    for tok in row.address_tokens:
        positions |= index.get((country, f"addr:{tok}"), set())
    for pin in row.postal_codes:
        positions |= index.get((country, f"pin:{pin}"), set())
    return positions


PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source", "similarity"]


def _checkpoint_state(checkpoint_dir: Path) -> dict:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return {"next_index": 0, "chunks": []}
    return json.loads(state_path.read_text())


def _write_checkpoint_state(checkpoint_dir: Path, next_index: int, chunks: list) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "state.json").write_text(json.dumps({"next_index": next_index, "chunks": chunks}))


def _flush_chunk(checkpoint_dir: Path, records: list, chunk_idx: int, next_index: int, chunks: list) -> list:
    """Write ``records`` (accumulated since the last flush) to a new chunk file and
    update the checkpoint state. Returns the updated ``chunks`` list. Caller clears
    ``records`` after this returns — this function never holds more than one flush
    interval's worth of rows at a time, which is what keeps peak memory bounded on a
    run producing tens of millions of candidate pairs over several hours."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = f"chunk_{chunk_idx:05d}.parquet"
    pd.DataFrame(records, columns=PAIR_COLUMNS).to_parquet(checkpoint_dir / chunk_name, index=False)
    chunks = chunks + [chunk_name]
    _write_checkpoint_state(checkpoint_dir, next_index, chunks)
    return chunks


def _merge_chunks(checkpoint_dir: Path, chunks: list, out_path: Path) -> None:
    """Stream-merge chunk parquet files into one final file without ever holding the
    full (potentially 10s of millions of rows) combined dataset in memory at once."""
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for chunk_name in chunks:
            table = pq.read_table(checkpoint_dir / chunk_name)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:  # no chunks at all -> write an empty file with the right schema
        pd.DataFrame(columns=PAIR_COLUMNS).to_parquet(out_path, index=False)


def clear_checkpoint(checkpoint_dir: Path) -> None:
    """Delete a completed run's checkpoint chunks/state (call after the final merged
    output has been written and verified)."""
    if not checkpoint_dir.exists():
        return
    for f in checkpoint_dir.iterdir():
        f.unlink()
    checkpoint_dir.rmdir()


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    source_frames: dict[str, pd.DataFrame],
    top_k: int = TOP_K_PER_SOURCE,
    min_similarity: float = MIN_NAME_SIMILARITY,
    show_progress: bool = True,
    checkpoint_dir: Path | None = None,
    checkpoint_interval_seconds: int = 300,
) -> pd.DataFrame:
    """Generate blocking candidates for every Source 1 entity.

    Parameters
    ----------
    s1_df: prepared (``prepare_records``) Source 1 dataframe.
    source_frames: mapping like ``{"S2": prepared_source2_df, "S3": prepared_source3_df}``.
    top_k: max candidates kept per (Source 1 entity, other source) pair.
    min_similarity: candidates below this cosine similarity are dropped, except the
        single best match for a source is always kept if the token/phonetic index
        found *any* overlap (keeps recall for weakly-similar but plausible matches).
    checkpoint_dir: when given, progress is flushed to disk every
        ``checkpoint_interval_seconds`` (as small parquet chunks, never holding the
        whole run's output in memory at once) and a re-run with the same
        ``checkpoint_dir`` resumes from the last flush instead of restarting — this
        run is hours long on the full test set, so losing it to a crash/interrupt and
        starting over is exactly what checkpointing avoids. On a clean finish, the
        chunks are left on disk; call :func:`clear_checkpoint` once you've verified
        the final merged output (the caller does this, not this function).

    Returns
    -------
    Long-format dataframe with columns: source1_entity_id, candidate_entity_id,
    source, similarity.
    """
    def _fit_tfidf(field: str):
        corpus = pd.concat(
            [s1_df[field]] + [f[field] for f in source_frames.values()],
            ignore_index=True,
        )
        # HashingVectorizer + TfidfTransformer, not TfidfVectorizer: TfidfVectorizer
        # (via CountVectorizer) builds one Python dict of every distinct n-gram across
        # the WHOLE corpus before applying max_features, which OOMs well before that
        # cap ever kicks in once the corpus reaches millions of documents (the full
        # test set: ~11.7M name/address strings). HashingVectorizer hashes n-grams
        # straight into a fixed-width feature space with no global vocabulary, so
        # memory stays bounded regardless of corpus size.
        hasher = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=2**20,
            alternate_sign=False, norm=None, dtype=np.float32,
        )
        counts = hasher.transform(corpus)
        mat = TfidfTransformer().fit_transform(counts).astype(np.float32)
        n_s1 = len(s1_df)
        s1_mat = mat[:n_s1].tocsr()
        offset = n_s1
        by_source = {}
        for name, frame in source_frames.items():
            by_source[name] = mat[offset : offset + len(frame)].tocsr()
            offset += len(frame)
        return s1_mat, by_source

    # Two independent similarity spaces: a true match may agree strongly on name
    # (reworded/typo'd address) OR on address (very different trade name), so
    # candidates are ranked by whichever signal is stronger, not name alone.
    _log("fitting name TF-IDF...")
    s1_name_tfidf, source_name_tfidf = _fit_tfidf("name_core")
    _log("fitting address TF-IDF...")
    s1_addr_tfidf, source_addr_tfidf = _fit_tfidf("address_clean")

    entity_ids = {name: frame["entity_id"].to_numpy() for name, frame in source_frames.items()}
    _log("building inverted index...")
    indices = {}
    for name, frame in source_frames.items():
        _log(f"  indexing source {name} ({len(frame):,} rows)...")
        indices[name] = _build_index(frame)
    rng = np.random.default_rng(RANDOM_SEED)

    n_s1 = len(s1_df)
    start_index, chunks = 0, []
    if checkpoint_dir is not None:
        state = _checkpoint_state(checkpoint_dir)
        start_index, chunks = state["next_index"], state["chunks"]
        if start_index:
            _log(f"resuming from checkpoint: {start_index:,}/{n_s1:,} entities already done "
                 f"({len(chunks)} chunk file(s) on disk)")

    records = []
    chunk_idx = len(chunks)
    last_flush = time.time()

    remaining = s1_df.iloc[start_index:]
    iterator = zip(range(start_index, n_s1), remaining.itertuples(index=False))
    if show_progress:
        iterator = tqdm(iterator, total=n_s1, initial=start_index, desc="blocking")

    def _flush_now(next_index):
        nonlocal records, chunk_idx, chunks, last_flush
        if checkpoint_dir is not None:
            chunk_idx += 1
            chunks = _flush_chunk(checkpoint_dir, records, chunk_idx, next_index, chunks)
        records = []
        last_flush = time.time()

    s1_pos = start_index - 1  # so the except-block reference below is always defined
    try:
        for s1_pos, row in iterator:
            s1_name_vec = s1_name_tfidf[s1_pos]
            s1_addr_vec = s1_addr_tfidf[s1_pos]
            for source_name in source_frames:
                candidate_positions = _candidate_positions(row, indices[source_name])
                if not candidate_positions:
                    continue
                positions = np.fromiter(candidate_positions, dtype=np.int64, count=len(candidate_positions))
                if len(positions) > MAX_CANDIDATE_POOL_SIZE:
                    positions = rng.choice(positions, size=MAX_CANDIDATE_POOL_SIZE, replace=False)

                name_sims = source_name_tfidf[source_name][positions] @ s1_name_vec.T
                name_sims = np.asarray(name_sims.todense()).ravel()
                addr_sims = source_addr_tfidf[source_name][positions] @ s1_addr_vec.T
                addr_sims = np.asarray(addr_sims.todense()).ravel()
                sims = np.maximum(name_sims, addr_sims)

                # Keep the top-K by name AND (separately) the top-K by address, unioned,
                # rather than one shared top-K ranked by max(name, addr). A shared ranking
                # lets "same address, different business" confusers (e.g. mall neighbors)
                # crowd true name-matches out of the cutoff; separate slates protect each
                # signal's own best candidates.
                name_order = np.argsort(-name_sims)[:top_k]
                addr_order = np.argsort(-addr_sims)[:top_k]
                order = np.union1d(name_order, addr_order)
                if len(order) == 0:
                    continue
                best_sim = sims[order].max()
                ids = entity_ids[source_name][positions[order]]
                for rank_pos, entity_id in zip(order, ids):
                    sim = sims[rank_pos]
                    if sim < min_similarity and sim < best_sim:
                        continue
                    records.append((row.entity_id, entity_id, source_name, float(sim)))

            if checkpoint_dir is not None and time.time() - last_flush > checkpoint_interval_seconds:
                _flush_now(s1_pos + 1)
    except (KeyboardInterrupt, Exception):
        if checkpoint_dir is not None and records:
            _log("interrupted — flushing checkpoint before exit...")
            resume_at = s1_pos + 1
            _flush_now(resume_at)
            _log(f"checkpoint saved ({chunks[-1]}) — re-run the same command to resume "
                 f"from entity {resume_at:,}/{n_s1:,}.")
        raise

    if checkpoint_dir is not None:
        if records:
            _flush_now(n_s1)
        _log(f"merging {len(chunks)} checkpoint chunk(s) into final result...")
        del indices, source_name_tfidf, source_addr_tfidf, s1_name_tfidf, s1_addr_tfidf
        gc.collect()
        tmp_out = checkpoint_dir / "_merged.parquet"
        _merge_chunks(checkpoint_dir, chunks, tmp_out)
        result = pd.read_parquet(tmp_out)
        tmp_out.unlink()
        return _finalize_pairs(result)

    return _finalize_pairs(pd.DataFrame(records, columns=PAIR_COLUMNS))


def _finalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast dtypes — at tens of millions of rows, float64 similarity and an
    object-dtype 'source' column (only ever "S2"/"S3") are pure waste."""
    df["similarity"] = df["similarity"].astype(np.float32)
    df["source"] = df["source"].astype("category")
    return df


def candidates_to_wide(pairs_df: pd.DataFrame, required_s1_ids) -> dict:
    """Collapse long-format candidate pairs into {s1_id: [candidate_ids]} for every
    required Source 1 id (entities with no candidates get an empty list)."""
    grouped = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(list)
    return {s1_id: grouped.get(s1_id, []) for s1_id in required_s1_ids}
