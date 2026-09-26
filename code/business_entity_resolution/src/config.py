"""Paths and shared constants for the entity resolution pipeline.

All paths are resolved relative to this file so scripts work regardless of the
caller's current working directory.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent               # code/business_entity_resolution
CODE_DIR = PIPELINE_DIR.parent              # code
REPO_ROOT = CODE_DIR.parent                 # amazon_ml

DATASET_DIR = REPO_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

OUTPUT_DIR = REPO_ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

MODELS_DIR = PIPELINE_DIR / "models"        # gitignored, holds trained model artifacts
FEATURES_DIR = PIPELINE_DIR / "features"    # gitignored, holds cached feature parquet files
CANDIDATES_DIR = PIPELINE_DIR / "candidates"  # gitignored, holds intermediate candidate-pair files

MATCHER_MODEL_PATH = MODELS_DIR / "matcher.json"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.json"
THRESHOLD_PATH = MODELS_DIR / "decision_threshold.json"

# --- Blocking ---
# Tokens appearing in more than this fraction of a source's records are treated as
# stopwords for inverted-index blocking (legal suffixes, generic words like "the").
MAX_TOKEN_DOC_FREQ = 0.02
MIN_TOKEN_LEN = 2
# Per-source cap on candidates kept per Source 1 entity after TF-IDF re-ranking.
TOP_K_PER_SOURCE = 20
MIN_NAME_SIMILARITY = 0.08

# --- Matching ---
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2  # fraction of Source 1 train entities held out for validation
NEGATIVE_SAMPLES_PER_POSITIVE = 5  # cap on non-match candidates kept per S1 entity for training

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp",
    "ltd", "limited", "pvt", "private", "plc", "lp", "gmbh", "sa", "sas", "sarl",
    "pllc", "pc", "pty", "bv", "nv", "kg", "ag",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "hwy": "highway", "apt": "apartment", "ste": "suite", "bldg": "building",
    "fl": "floor", "no": "number", "pkwy": "parkway", "sq": "square",
}
