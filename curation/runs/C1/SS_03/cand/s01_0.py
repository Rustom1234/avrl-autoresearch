"""Curator: keep only clean, deduplicated, target-style (WikiText-like) documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. damage filter : drop docs < 200 chars and docs with homoglyph-mixed words
  2. bigram score  : fraction of the first 600 word bigrams seen in target_sample,
                     thresholded at 0.75 x the 5th percentile of held-out target scores
  3. style score   : starts with "= " or has enough spaced punctuation
  4. dedupe        : exact (whitespace-normalised) and near (first 300 chars)
  5. seeded shuffle of the survivors; no filler is added

Notes on choices made here:
  - Dedupe runs after the quality filters, so a damaged copy cannot take the
    "first occurrence" slot and cause a clean later copy to be dropped.
  - The threshold is calibrated on 100-doc halves but pool docs are scored
    against all 200 target docs, so the threshold errs on the lenient side.
  - No token count is computed: nothing is truncated or padded, so the
    len(text)/4 estimate would not change the output.
"""
import hashlib
import os
import random
import re

import numpy as np

MIN_CHARS = 200
MAX_MIXED_FRAC = 0.005
N_BIGRAMS = 600
NEAR_DUP_CHARS = 300

_WORD_RE = re.compile(r"[^\W\d_]+")
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
# Greek, Greek Extended, Cyrillic (+ supplement), Armenian
_OTHER_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff\u0400-\u052f\u0531-\u058f]")


def _is_homoglyph_damaged(text):
    if _OTHER_RE.search(text) is None:
        return False
    words = _WORD_RE.findall(text)
    if not words:
        return False
    mixed = 0
    for w in words:
        if _OTHER_RE.search(w) is not None and _LATIN_RE.search(w) is not None:
            mixed += 1
    return mixed > MAX_MIXED_FRAC * len(words)


def _head_words(text):
    """First N_BIGRAMS + 1 lowercased whitespace tokens."""
    parts = text.split(None, N_BIGRAMS + 1)[: N_BIGRAMS + 1]
    return [p.lower() for p in parts]


def _all_bigrams(text):
    w = text.lower().split()
    return set(zip(w, w[1:]))


def _bigram_hit(text, bigram_set):
    w = _head_words(text)
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for bg in zip(w, w[1:]):
        if bg in bigram_set:
            hits += 1
    return hits / n


def _punct_rate(text):
    n_words = len(text.split())
    if n_words == 0:
        return 0.0
    c = text.count(" , ") + text.count(" . ") + text.count(" @-@ ")
    return c / n_words


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    target_sample = [t for t in target_sample if isinstance(t, str)]

    # Target bigram set and leave-out calibration of the threshold.
    full_set = set()
    for t in target_sample:
        full_set |= _all_bigrams(t)

    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    held_out = []
    if half_a and half_b:
        set_a = set()
        for t in half_a:
            set_a |= _all_bigrams(t)
        set_b = set()
        for t in half_b:
            set_b |= _all_bigrams(t)
        held_out.extend(_bigram_hit(t, set_a) for t in half_b)
        held_out.extend(_bigram_hit(t, set_b) for t in half_a)
    h_thresh = 0.75 * float(np.percentile(held_out, 5)) if held_out else 0.0

    # Style threshold: half the target's 5th percentile spaced-punctuation rate.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
    else:
        p_thresh = 0.0

    seen_exact = set()
    seen_near = set()
    keep = []

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue

        # Word-order / domain score.
        if _bigram_hit(text, full_set) < h_thresh:
            continue

        # Style score.
        if not text.lstrip().startswith("= ") and _punct_rate(text) < p_thresh:
            continue

        # Homoglyph damage.
        if _is_homoglyph_damaged(text):
            continue

        # Dedupe among survivors, first occurrence wins.
        norm = " ".join(text.split())
        d_exact = _digest(norm)
        if d_exact in seen_exact:
            continue
        d_near = _digest(norm[:NEAR_DUP_CHARS])
        if d_near in seen_near:
            continue
        seen_exact.add(d_exact)
        seen_near.add(d_near)

        keep.append(i)

    if not keep:
        # Safety net: never return an empty list.
        keep = list(range(len(docs)))

    random.Random(seed).shuffle(keep)
    return keep
