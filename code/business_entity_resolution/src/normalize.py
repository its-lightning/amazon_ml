"""Text normalization for business names and addresses.

Handles the noise patterns called out in the challenge brief: punctuation/casing
differences, legal-suffix variants, transliteration (e.g. Devanagari-script Indian
business names), address abbreviations, and stray leading/trailing junk characters
(e.g. "-- Holloway Peak Inc", "<< Team Ecole").
"""

import re

from anyascii import anyascii

from .config import ADDRESS_ABBREVIATIONS, LEGAL_SUFFIXES, MIN_TOKEN_LEN

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")
_POSTAL_CODE = re.compile(r"\b\d{5,6}\b")


def to_ascii_lower(text: str) -> str:
    """Transliterate to ASCII (handles Devanagari and other scripts) and lowercase."""
    if not text:
        return ""
    return anyascii(text).lower()


def clean_text(text: str) -> str:
    """Transliterate, drop punctuation, collapse whitespace, strip stray junk."""
    text = to_ascii_lower(text)
    text = text.replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return [t for t in text.split(" ") if len(t) >= MIN_TOKEN_LEN]


def normalize_name(raw_name: str) -> dict:
    """Return normalized forms of a business name used across blocking/features.

    Keys:
        clean: full cleaned name (ascii, lowercase, no punctuation)
        tokens: whitespace tokens of ``clean``
        core_tokens: ``tokens`` with legal suffixes (Inc/Corp/Pvt Ltd/...) removed
        core: ``core_tokens`` joined back into a string
    """
    clean = clean_text(raw_name)
    tokens = tokenize(clean)
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = tokens
    return {
        "clean": clean,
        "tokens": tokens,
        "core_tokens": core_tokens,
        "core": " ".join(core_tokens),
    }


def normalize_address(raw_address: str) -> dict:
    """Return normalized forms of an address used across blocking/features.

    Keys:
        clean: full cleaned address with common abbreviations expanded
        tokens: whitespace tokens of ``clean``
        postal_codes: set of 5-6 digit numeric tokens found (PIN/ZIP candidates)
    """
    clean = clean_text(raw_address)
    expanded_tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in clean.split(" ") if t]
    clean = " ".join(expanded_tokens)
    tokens = tokenize(clean)
    postal_codes = set(_POSTAL_CODE.findall(raw_address))
    return {
        "clean": clean,
        "tokens": tokens,
        "postal_codes": postal_codes,
    }
