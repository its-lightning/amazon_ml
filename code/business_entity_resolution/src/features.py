"""Pairwise similarity features for candidate (Source 1, candidate) pairs.

Used both to build the training feature table (joined against ground truth labels)
and to score test-set candidates at inference time. All features are computed from
the two normalized records only — no external data.
"""

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "name_tfidf_cosine",
    "name_levenshtein_ratio",
    "name_token_jaccard",
    "common_token_count",
    "name_len_diff_ratio",
    "address_levenshtein_ratio",
    "address_token_jaccard",
    "postal_code_known",
    "postal_code_match",
    "phonetic_match",
    "exact_core_name_match",
]


def _pairwise_features(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned feature computation between two same-length record frames."""
    n = len(a)
    out = {col: np.zeros(n, dtype=np.float32) for col in FEATURE_COLUMNS if col != "name_tfidf_cosine"}

    a_core = a["name_core"].to_numpy()
    b_core = b["name_core"].to_numpy()
    a_addr = a["address_clean"].to_numpy()
    b_addr = b["address_clean"].to_numpy()
    a_tokens = a["core_tokens"].to_numpy()
    b_tokens = b["core_tokens"].to_numpy()
    a_atokens = a["address_tokens"].to_numpy()
    b_atokens = b["address_tokens"].to_numpy()
    a_postal = a["postal_codes"].to_numpy()
    b_postal = b["postal_codes"].to_numpy()

    for i in range(n):
        na, nb = a_core[i], b_core[i]
        out["name_levenshtein_ratio"][i] = fuzz.ratio(na, nb) / 100.0
        len_a, len_b = len(na), len(nb)
        out["name_len_diff_ratio"][i] = abs(len_a - len_b) / max(len_a, len_b, 1)
        out["exact_core_name_match"][i] = 1.0 if na and na == nb else 0.0

        ta, tb = set(a_tokens[i]), set(b_tokens[i])
        union = ta | tb
        common = ta & tb
        out["name_token_jaccard"][i] = len(common) / len(union) if union else 0.0
        out["common_token_count"][i] = len(common)

        out["address_levenshtein_ratio"][i] = fuzz.ratio(a_addr[i], b_addr[i]) / 100.0
        aa, ab = set(a_atokens[i]), set(b_atokens[i])
        aunion = aa | ab
        out["address_token_jaccard"][i] = len(aa & ab) / len(aunion) if aunion else 0.0

        pa, pb = a_postal[i], b_postal[i]
        if pa and pb:
            out["postal_code_known"][i] = 1.0
            out["postal_code_match"][i] = 1.0 if (pa & pb) else 0.0

        # Blocking already guarantees every candidate shares >=1 exact token or
        # phonetic code overall, so a naive overlap check here would be constant.
        # Instead: do the *non-shared* tokens on each side still sound alike
        # (typo/spelling variant), beyond what common_token_count already captures?
        a_excl, b_excl = ta - common, tb - common
        a_ph_excl = {jellyfish.metaphone(t) for t in a_excl}
        b_ph_excl = {jellyfish.metaphone(t) for t in b_excl}
        out["phonetic_match"][i] = 1.0 if (a_ph_excl & b_ph_excl) else 0.0

    return pd.DataFrame(out)


def compute_pair_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    source_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute :data:`FEATURE_COLUMNS` for every row of ``pairs_df``.

    ``pairs_df`` must have columns source1_entity_id, candidate_entity_id, source,
    similarity (as produced by :func:`blocking.generate_candidate_pairs`).
    """
    s1_idx = s1_df.set_index("entity_id")
    source_idx = {name: frame.set_index("entity_id") for name, frame in source_frames.items()}

    chunks = []
    for source_name, group in pairs_df.groupby("source", sort=False):
        other_idx = source_idx[source_name]
        a = s1_idx.loc[group["source1_entity_id"]].reset_index(drop=True)
        b = other_idx.loc[group["candidate_entity_id"]].reset_index(drop=True)
        feats = _pairwise_features(a, b)
        feats["name_tfidf_cosine"] = group["similarity"].to_numpy(dtype=np.float32)
        feats["source1_entity_id"] = group["source1_entity_id"].to_numpy()
        feats["candidate_entity_id"] = group["candidate_entity_id"].to_numpy()
        feats["source"] = source_name
        chunks.append(feats)

    result = pd.concat(chunks, ignore_index=True)
    result["source"] = result["source"].astype("category")
    ordered_cols = ["source1_entity_id", "candidate_entity_id", "source"] + FEATURE_COLUMNS
    return result[ordered_cols]
