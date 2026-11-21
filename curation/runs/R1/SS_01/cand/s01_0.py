"""Curator: dedupe + word-bigram target-likeness filter calibrated on the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SCORE_TOKENS = 2000      # word tokens per doc used for scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02         # keep slightly more than T so the cap never causes cycling
TIKTOKEN_DEADLINE = 100  # seconds since start after which we stop exact tokenising

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}


def _head_tokens(text, n=SCORE_TOKENS):
    """First n lowercase whitespace tokens."""
    return [t.lower() for t in text.split(None, n)[:n]]


def _clean_frac(text):
    if not text:
        return 0.0
    bad = len(text.translate(_DELETE_ALLOWED))
    return 1.0 - bad / len(text)


def _hit_rate(toks, contains):
    n = len(toks) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(toks, toks[1:]):
        if contains(b):
            hits += 1
    return hits / n


def _count_tokens(texts, t0):
    """gpt2 token counts; falls back to len/4 if tiktoken is unavailable or slow."""
    counts = [max(1, len(t) // 4) for t in texts]
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        chunk = 512
        for s in range(0, len(texts), chunk):
            if time.time() - t0 > TIKTOKEN_DEADLINE:
                break
            batch = enc.encode_ordinary_batch(texts[s:s + chunk])
            for j, ids in enumerate(batch):
                counts[s + j] = max(1, len(ids))
    except Exception:
        pass
    return counts


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- 2. reference bigram set + 4. leave-one-out calibration -------------
    df = Counter()            # number of target docs containing each bigram
    target_heads = []
    for t in target_sample:
        toks = t.lower().split()
        df.update(set(zip(toks, toks[1:])))
        target_heads.append(toks[:SCORE_TOKENS])
    B = set(df)

    loo = []
    for toks in target_heads:
        if len(toks) < 2:
            continue
        # bigram is in the other 199 docs iff its doc-frequency is >= 2
        loo.append(_hit_rate(toks, lambda b: df.get(b, 0) >= 2))
    threshold = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    threshold = max(threshold, 0.0)

    # ---- 1. dedupe, 3. score, 5. junk filter --------------------------------
    seen_full = set()
    seen_prefix = set()
    cand = []                 # (score, idx) for unique, non-junk docs
    b_contains = B.__contains__
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        norm = " ".join(text.lower().split())
        if not norm:
            continue
        h = hash(norm)
        p = hash(norm[:PREFIX_CHARS])
        if h in seen_full or p in seen_prefix:
            continue
        seen_full.add(h)
        seen_prefix.add(p)

        toks = _head_tokens(text)
        if len(toks) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue
        cand.append((_hit_rate(toks, b_contains), i))

    if not cand:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # best first, deterministic tie-break on index
    cand.sort(key=lambda x: (-x[0], x[1]))
    kept = [i for s, i in cand if s >= threshold]

    # ---- 6. cap at ~T tokens, keeping highest-scoring ------------------------
    if kept:
        counts = _count_tokens([docs[i] for i in kept], t0)
        budget = T * OVERSHOOT
        total = 0
        out = []
        for i, c in zip(kept, counts):
            if total >= budget:
                break
            out.append(i)
            total += c
        kept = out
    else:
        # Nothing passed the threshold: fall back to the best-scoring clean docs.
        budget = T * OVERSHOOT
        total = 0
        for s, i in cand:
            if total >= budget:
                break
            kept.append(i)
            total += max(1, len(docs[i]) // 4)

    # ---- 7. seeded shuffle ---------------------------------------------------
    rng.shuffle(kept)
    return kept
