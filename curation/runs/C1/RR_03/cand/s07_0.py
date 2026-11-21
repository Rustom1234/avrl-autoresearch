"""Curator: keep clean, intact, in-domain (WikiText-style) documents, de-duplicated,
in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_WORDS = 1500          # words used for the bigram score
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")
MIN_UNIQUE_RATIO = 0.45
MAX_SINGLE_CHAR_FRAC = 0.10


def _words(text):
    return text.lower().split()


def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)[:MAX_WORDS + 1]
        B.update(zip(w, w[1:]))
    return B


def _score_words(words, B):
    """Fraction of the bigrams in the first MAX_WORDS words that are in B."""
    w = words[:MAX_WORDS]
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(w, w[1:]):
        if b in B:
            hits += 1
    return hits / n


def _calibrate_tau(target_sample):
    """tau = 0.8 x 5th percentile of two-fold held-out target scores."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half = n // 2
    a, b = target_sample[:half], target_sample[half:]
    scores = []
    for build, held in ((a, b), (b, a)):
        B = _bigram_set(build)
        for t in held:
            w = _words(t)
            if len(w) >= 2:
                scores.append(_score_words(w, B))
    if not scores:
        return 0.0
    return 0.8 * float(np.percentile(np.asarray(scores), 5))


def _nonascii_letter_frac(text):
    n = len(text)
    if n == 0:
        return 0.0
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii <= MAX_NONASCII_LETTER_FRAC * n:
        return 0.0
    cnt = sum(1 for c in text if ord(c) > 127 and c.isalpha())
    return cnt / n


def _repetitive(words):
    n = len(words) - 4
    if n <= 0:
        return False
    top = Counter(
        zip(words, words[1:], words[2:], words[3:], words[4:])
    ).most_common(1)[0][1]
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n


def _check_info_density(words):
    """Rejects documents based on unique word ratio and single-character concentration."""
    n = len(words)
    if n == 0:
        return False, False
    
    unique_count = len(set(words))
    unique_ratio = unique_count / n
    
    single_char_count = sum(1 for w in words if len(w) == 1)
    single_char_frac = single_char_count / n
    
    # Return (is_low_density, is_high_single_char_frac)
    return unique_ratio < MIN_UNIQUE_RATIO, single_char_frac > MAX_SINGLE_CHAR_FRAC


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    if target_sample:
        hdr_frac = sum(
            1 for t in target_sample if t.lstrip().startswith("= ")
        ) / len(target_sample)
    else:
        hdr_frac = 0.0
    require_header = hdr_frac >= 0.9

    seen_full = set()
    seen_prefix = set()
    kept = []
    reasons = Counter()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            reasons["short"] += 1
            continue
        if require_header and not text.lstrip().startswith("= "):
            reasons["no_header"] += 1
            continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC:
            reasons["non_ascii"] += 1
            continue

        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS):
            reasons["spam"] += 1
            continue

        words = low.split()
        
        # Information Density Filters
        low_density, high_single_char = _check_info_density(words)
        if low_density:
            reasons["low_density"] += 1
            continue
        if high_single_char:
            reasons["high_single_char"] += 1
            continue

        if _score_words(words, B) < tau:
            reasons["low_score"] += 1
            continue
        if _repetitive(words):
            reasons["repetition"] += 1
            continue

        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full or prefix in seen_prefix:
            reasons["duplicate"] += 1
            continue
        seen_full.add(norm)
        seen_prefix.add(prefix)
        kept.append(i)

    if not kept:
        print("[curate] WARNING: filter kept nothing; falling back to raw shuffle")
        kept = list(range(len(docs)))

    random.Random(seed).shuffle(kept)

    est_tokens = sum(len(docs[i]) for i in kept) // 4
    tok_kind = "chars/4"
    if time.time() - t0 < 60:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            est_tokens = sum(
                len(x) for x in enc.encode_ordinary_batch([docs[i] for i in kept])
            )
            tok_kind = "tiktoken"
        except Exception:
            pass

    print(
        "[curate] pool=%d kept=%d kept_tokens~%d (%s, %.2fx T) tau=%.4f "
        "header_rule=%s dropped=%s time=%.1fs"
        % (
            len(docs),
            len(kept),
            est_tokens,
            tok_kind,
            est_tokens / float(T),
            tau,
            require_header,
            dict(reasons),
            time.time() - t0,
        )
    )
    return kept
