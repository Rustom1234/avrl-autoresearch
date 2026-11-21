"""Curator: drop junk and duplicates, then seeded random shuffle of the survivors.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import hashlib
import os
import random

BANNED_PHRASES = ("Click here", "discount code", "coupon")
MAX_NON_ASCII_FRAC = 0.05


def _is_junk(text):
    if any(p in text for p in BANNED_PHRASES):
        return True
    if text:
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        if non_ascii > MAX_NON_ASCII_FRAC * len(text):
            return True
    return False


def _norm_hash(text):
    norm = " ".join(text.split()).lower()
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def select(docs, target_sample):
    seen = set()
    kept = []
    for i, text in enumerate(docs):
        if _is_junk(text):
            continue
        h = _norm_hash(text)
        if h in seen:
            continue
        seen.add(h)
        kept.append(i)
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(kept)
    return kept
