"""Curator: dedupe + junk filter + word-order-sensitive ranking against the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact duplicates (whitespace-normalised text) and near-duplicates
     (first 300 normalised characters), keeping the first occurrence.
  2. Hard junk filters (too short, too few ASCII letters, spam templates).
  3. Score each document on its first ~2000 whitespace tokens:
         score = bigram_hit / (unigram_hit + 0.05) + 0.5 * bigram_hit (+0.1 for a '= Title =' header)
     Word-shuffled articles have high unigram_hit but low bigram_hit, so they score low.
  4. Threshold = 5th percentile of leave-one-out scores of the target sample.
  5. Take documents in descending score order until about T tokens are collected
     (everything above the threshold if that fits, topped up from below it if needed).
  6. Shuffle the selection with the seed.
"""
import os
import random
import re
import time
from collections import Counter

import numpy as np

from common import T

MAX_WORDS = 2000
NEAR_DUP_CHARS = 300
MIN_CHARS = 200
MAX_NON_LETTER_RATIO = 0.4
HEADER_BONUS = 0.1
THRESHOLD_PERCENTILE = 5
TIME_BUDGET_FOR_TOKENISING = 120.0  # seconds since start; after this, fall back to len/4

SPAM_TEMPLATES = (
    "discount code |",
    "online store |",
    "Click here to order",
    "free shipping on all",
)

_ASCII_LETTERS_RE = re.compile(r"[A-Za-z]+")


def _first_words(text):
    return text.split(None, MAX_WORDS)[:MAX_WORDS]


def _has_header(text):
    t = text.lstrip()
    if not t.startswith("= "):
        return False
    first_line = t.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =") and len(first_line) > 3


def _combine(bigram_hit, unigram_hit, header):
    score = bigram_hit / (unigram_hit + 0.05) + 0.5 * bigram_hit
    if header:
        score += HEADER_BONUS
    return score


def _score(text, target_unigrams, target_bigrams):
    words = _first_words(text)
    n = len(words)
    if n < 2:
        return 0.0
    uni_hits = sum(1 for w in words if w in target_unigrams)
    bi_hits = sum(1 for b in zip(words, words[1:]) if b in target_bigrams)
    return _combine(bi_hits / (n - 1), uni_hits / n, _has_header(text))


def _is_junk(text):
    stripped = text.strip()
    n = len(stripped)
    if n < MIN_CHARS:
        return True
    non_letters = len(_ASCII_LETTERS_RE.sub("", stripped))
    if non_letters / n > MAX_NON_LETTER_RATIO:
        return True
    for tpl in SPAM_TEMPLATES:
        if tpl in text:
            return True
    return False


def _calibrate_threshold(target_sample):
    """Leave-one-out scores of the target documents; return (threshold, unigrams, bigrams)."""
    uni_df = Counter()
    bi_df = Counter()
    per_doc = []
    for text in target_sample:
        words = text.split()
        u = set(words)
        b = set(zip(words, words[1:]))
        uni_df.update(u)
        bi_df.update(b)
        per_doc.append((u, b))

    loo_scores = []
    for text in target_sample:
        words = _first_words(text)
        n = len(words)
        if n < 2:
            continue
        # A unigram/bigram is in the "other 199" sets iff its document frequency
        # exceeds 1 (this document itself always contributes exactly 1).
        uni_hits = sum(1 for w in words if uni_df[w] > 1)
        bi_hits = sum(1 for bg in zip(words, words[1:]) if bi_df[bg] > 1)
        loo_scores.append(_combine(bi_hits / (n - 1), uni_hits / n, _has_header(text)))

    if loo_scores:
        threshold = float(np.percentile(np.asarray(loo_scores), THRESHOLD_PERCENTILE))
    else:
        threshold = 0.0
    return threshold, set(uni_df), set(bi_df)


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # --- 1. dedupe (exact + near), 4. hard junk filters -------------------
    seen_exact = set()
    seen_prefix = set()
    survivors = []
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        norm = " ".join(text.split())
        h = hash(norm)
        if h in seen_exact:
            continue
        p = hash(norm[:NEAR_DUP_CHARS])
        if p in seen_prefix:
            continue
        seen_exact.add(h)
        seen_prefix.add(p)
        if _is_junk(text):
            continue
        survivors.append(i)

    # --- 2. target sets, 5. threshold --------------------------------------
    threshold, target_unigrams, target_bigrams = _calibrate_threshold(target_sample)

    # --- 3. score ------------------------------------------------------------
    scores = {i: _score(docs[i], target_unigrams, target_bigrams) for i in survivors}

    # Descending score, ties broken by pool position (deterministic).
    ranked = sorted(survivors, key=lambda i: (-scores[i], i))

    # --- 6. select up to about T tokens --------------------------------------
    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0
    pos = 0
    batch = 256
    n_ranked = len(ranked)
    while pos < n_ranked and total < T:
        chunk = ranked[pos:pos + batch]
        pos += batch
        counts = None
        if enc is not None and (time.time() - t0) < TIME_BUDGET_FOR_TOKENISING:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in chunk])]
            except Exception:
                counts = None
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in chunk]
        for i, c in zip(chunk, counts):
            selected.append(i)
            total += c
            if total >= T:
                break
    # Note: walking the ranking until T is reached covers both cases:
    #   - above-threshold docs exceed T  -> only the top-scoring ones are kept;
    #   - above-threshold docs fall short -> the next-highest survivors are appended.
    # `threshold` therefore only matters as a diagnostic of where the cut falls.
    _ = threshold

    if not selected:
        # Degenerate pool: fall back to the raw behaviour rather than return nothing.
        selected = list(range(len(docs)))

    # --- 7. order ------------------------------------------------------------
    rng.shuffle(selected)
    return selected
