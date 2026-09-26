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

from collections import defaultdict

import jellyfish
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .config import (
    MAX_CANDIDATE_POOL_SIZE,
    MAX_TOKEN_DOC_FREQ,
    MIN_NAME_SIMILARITY,
    RANDOM_SEED,
    TOP_K_PER_SOURCE,
)
from .normalize import normalize_address, normalize_name


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

    for pos, row in enumerate(df.itertuples(index=False)):
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


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    source_frames: dict[str, pd.DataFrame],
    top_k: int = TOP_K_PER_SOURCE,
    min_similarity: float = MIN_NAME_SIMILARITY,
    show_progress: bool = True,
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
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=2**20, dtype=np.float32
        )
        mat = vec.fit_transform(corpus)
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
    s1_name_tfidf, source_name_tfidf = _fit_tfidf("name_core")
    s1_addr_tfidf, source_addr_tfidf = _fit_tfidf("address_clean")

    entity_ids = {name: frame["entity_id"].to_numpy() for name, frame in source_frames.items()}
    indices = {name: _build_index(frame) for name, frame in source_frames.items()}
    rng = np.random.default_rng(RANDOM_SEED)

    records = []
    iterator = s1_df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(s1_df), desc="blocking")

    for s1_pos, row in enumerate(iterator):
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

    return pd.DataFrame(records, columns=["source1_entity_id", "candidate_entity_id", "source", "similarity"])


def candidates_to_wide(pairs_df: pd.DataFrame, required_s1_ids) -> dict:
    """Collapse long-format candidate pairs into {s1_id: [candidate_ids]} for every
    required Source 1 id (entities with no candidates get an empty list)."""
    grouped = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(list)
    return {s1_id: grouped.get(s1_id, []) for s1_id in required_s1_ids}
