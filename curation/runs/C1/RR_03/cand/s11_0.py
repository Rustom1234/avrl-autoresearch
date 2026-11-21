"""Curator: keep clean, intact, in-domain (WikiText-style) documents, de-duplicated,
ranked by bigram score in two tiers, cut to the token budget T, then shuffled.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Rules (a document must pass all of them to be a candidate):
  - minimum length, non-ASCII letter fraction, spam patterns
  - bigram score >= calibrated tau
  - no heavy 5-gram repetition
  - semantic integrity: unique words <= 80% of total words (noise / scrambling)
  - semantic integrity: longest word <= 30 characters (URLs / base64)
Tiers:
  - Tier 1: passes all rules including the "= " header rule
  - Tier 2: passes all rules except the header rule
Selection: best-scoring Tier 1 docs first, then best-scoring Tier 2 docs, until the
token budget T is filled; the selected set is then shuffled with the seed.
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

MAX_UNIQUE_WORD_FRAC = 0.80   # semantic integrity: type/token ratio ceiling
MIN_WORDS_FOR_UNIQUE = 100    # short texts have a naturally high type/token ratio
MAX_WORD_LEN = 30             # semantic integrity: longest allowed word

TOKENIZE_CHUNK = 1000
TOKENIZE_DEADLINE = 140.0     # seconds; after this, fall back to chars/4


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
    # Fast pre-check: total non-ASCII characters.
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
    # require top >= 3 so that very short docs are not dropped trivially
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n


def _too_many_unique(words):
    """Unique words exceed 80% of total words (random noise / scrambled text)."""
    n = len(words)
    if n < MIN_WORDS_FOR_UNIQUE:
        return False
    return len(set(words)) > MAX_UNIQUE_WORD_FRAC * n


def _has_long_word(words):
    """Longest word is longer than MAX_WORD_LEN characters (URLs, base64, ...)."""
    if not words:
        return False
    return max(map(len, words)) > MAX_WORD_LEN


def _fill_budget(docs, ranked, t0):
    """Take docs in ranked order until their token total reaches T.

    Tokens are counted lazily (chunk by chunk, in rank order) with tiktoken; if
    tiktoken is unavailable or time runs short, chars/4 is used instead.
    Returns (selected indices, token total, counting method).
    """
    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0
    kind = "tiktoken" if enc is not None else "chars/4"
    pos = 0
    n = len(ranked)
    while pos < n and total < T:
        chunk = ranked[pos:pos + TOKENIZE_CHUNK]
        pos += len(chunk)
        counts = None
        if enc is not None and time.time() - t0 < TOKENIZE_DEADLINE:
            try:
                counts = [
                    len(x)
                    for x in enc.encode_ordinary_batch([docs[i] for i in chunk])
                ]
            except Exception:
                counts = None
        if counts is None:
            if kind == "tiktoken":
                kind = "tiktoken+chars/4"
            counts = [max(1, len(docs[i]) // 4) for i in chunk]
        for i, c in zip(chunk, counts):
            selected.append(i)
            total += c
            if total >= T:
                break
    return selected, total, kind


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    # Only use the header rule for tiering if the target sample really follows it.
    if target_sample:
        hdr_frac = sum(
            1 for t in target_sample if t.lstrip().startswith("= ")
        ) / len(target_sample)
    else:
        hdr_frac = 0.0
    require_header = hdr_frac >= 0.9

    reasons = Counter()
    # candidates[tier] = list of (index, score, full_hash, prefix_hash)
    candidates = {1: [], 2: []}

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            reasons["short"] += 1
            continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC:
            reasons["non_ascii"] += 1
            continue

        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS):
            reasons["spam"] += 1
            continue

        words = low.split()
        if _has_long_word(words):
            reasons["long_word"] += 1
            continue
        if _too_many_unique(words):
            reasons["unique_words"] += 1
            continue

        score = _score_words(words, B)
        if score < tau:
            reasons["low_score"] += 1
            continue
        if _repetitive(words):
            reasons["repetition"] += 1
            continue

        has_header = text.lstrip().startswith("= ")
        tier = 1 if (has_header or not require_header) else 2

        norm = " ".join(words)
        candidates[tier].append((i, score, hash(norm), hash(norm[:300])))

    # Dedup among surviving docs: Tier 1 first (so a headerless copy cannot shadow
    # an intact one), first occurrence wins within a tier.
    seen_full = set()
    seen_prefix = set()
    tiers = {1: [], 2: []}
    for tier in (1, 2):
        for i, score, hf, hp in candidates[tier]:
            if hf in seen_full or hp in seen_prefix:
                reasons["duplicate"] += 1
                continue
            seen_full.add(hf)
            seen_prefix.add(hp)
            tiers[tier].append((i, score))

    # Rank: Tier 1 by score (best first), then Tier 2 by score (best first).
    ranked = []
    for tier in (1, 2):
        ordered = sorted(tiers[tier], key=lambda x: (-x[1], x[0]))
        ranked.extend(i for i, _ in ordered)

    if not ranked:
        # Degenerate case: never return an empty list.
        print("[curate] WARNING: filter kept nothing; falling back to raw shuffle")
        kept = list(range(len(docs)))
        random.Random(seed).shuffle(kept)
        return kept

    kept, est_tokens, tok_kind = _fill_budget(docs, ranked, t0)
    n_tier1 = len(tiers[1])
    used_t1 = min(len(kept), n_tier1)
    used_t2 = len(kept) - used_t1

    random.Random(seed).shuffle(kept)

    print(
        "[curate] pool=%d tier1=%d tier2=%d kept=%d (t1=%d, t2=%d) "
        "kept_tokens~%d (%s, %.2fx T) tau=%.4f header_rule=%s dropped=%s time=%.1fs"
        % (
            len(docs),
            n_tier1,
            len(tiers[2]),
            len(kept),
            used_t1,
            used_t2,
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
