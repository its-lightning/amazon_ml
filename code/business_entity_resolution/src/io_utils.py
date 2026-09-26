"""Reading/writing the challenge's tab-separated files.

Every read/write here is explicit about the delimiter and about NOT letting pandas'
default NA-sniffing mangle fields (an address or ID that happens to read like "NA" or
an empty string must stay a literal string, not become NaN).
"""

from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path) -> pd.DataFrame:
    """Read one of the *_source1/2/3.tsv files."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}")
    return df


def read_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    df["matched_entity_ids"] = df["matched_entity_ids"].apply(
        lambda s: [x for x in s.split(",") if x] if s else []
    )
    return df


def write_id_list_tsv(path: Path, rows: dict, id_col: str, list_col: str) -> None:
    """Write a {source1_entity_id: iterable_of_ids} mapping in the required format.

    ``rows`` values may be any iterable of entity id strings (order is preserved via
    sorting for determinism); duplicates within a value are removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id in sorted(rows):
            ids = sorted(set(rows[s1_id]))
            f.write(f"{s1_id}\t{','.join(ids)}\n")
