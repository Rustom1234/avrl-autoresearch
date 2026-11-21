"""Curator: dedupe + word-bigram target-likeness filter, ordered so the most
target-like text is read last.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.

The filter (dedupe, bigram hit-rate score, leave-one-out threshold, junk filter)
selects a set of docs, capped by score at ~1.02*T tokens.  The set is split by
score into H (top docs, ~40% of the set's tokens, never more than 50% of T) and
L (the rest).  The returned sequence is built explicitly to T..1.02*T tokens:

    [early part: L shuffled, then extra shuffled passes over L+H if needed]
    + [final part: all of H, shuffled]

so the H block sits at the end of training and the trainer never wraps around.
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
OVERSHOOT = 1.02         # hard upper bound on sequence length, as a multiple of T
TARGET_TOTAL = 1.01      # aimed sequence length, as a multiple of T
H_FRAC = 0.40            # share of tokens placed in the final (most target-like) block
H_MAX_FRAC = 0.50        # the final block never exceeds this share of T
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


def _build_order(capped, tok, rng):
    """capped: doc indices, best score first. tok: idx -> token count."""
    total = sum(tok[i] for i in capped)

    # ---- split by score: H = top docs with ~H_FRAC of the tokens -------------
    h_target = H_FRAC * min(total, T)
    h_max = H_MAX_FRAC * T
    n_h = 0
    h_tok = 0
    for i in capped:
        c = tok[i]
        if h_tok >= h_target or h_tok + c > h_max:
            break
        h_tok += c
        n_h += 1
    H = capped[:n_h]
    L = capped[n_h:]

    e_min = T - h_tok                      # below this the trainer would wrap
    e_target = TARGET_TOTAL * T - h_tok    # aimed size of the early part
    e_max = OVERSHOOT * T - h_tok          # above this the total exceeds 1.02*T

    # If L is larger than needed, drop its lowest-scoring docs.
    l_tok = sum(tok[i] for i in L)
    while L and l_tok - tok[L[-1]] >= e_target:
        l_tok -= tok[L.pop()]

    early = []
    early_tok = 0

    def add(seq):
        nonlocal early_tok
        for i in seq:
            if early_tok >= e_target:
                return
            c = tok[i]
            if early_tok + c > e_max:
                continue
            early.append(i)
            early_tok += c

    # ---- early part: L shuffled, then extra shuffled passes over L+H ---------
    first = list(L)
    rng.shuffle(first)
    add(first)
    while early_tok < e_target:
        before = early_tok
        perm = list(capped)
        rng.shuffle(perm)
        add(perm)
        if early_tok == before:
            break

    # Safety net: never leave the sequence shorter than T.
    while early_tok < e_min:
        perm = list(capped)
        rng.shuffle(perm)
        for i in perm:
            if early_tok >= e_min:
                break
            early.append(i)
            early_tok += tok[i]

    # ---- final part: all of H, shuffled ---------------------------------------
    final = list(H)
    rng.shuffle(final)
    return early + final


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- reference bigram set + leave-one-out calibration --------------------
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

    # ---- dedupe, score, junk filter -------------------------------------------
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
    if not kept:
        # Nothing passed the threshold: fall back to the best-scoring clean docs.
        kept = [i for s, i in cand]

    # ---- token counts, cap at ~1.02*T keeping the highest-scoring -------------
    counts = _count_tokens([docs[i] for i in kept], t0)
    budget = T * OVERSHOOT
    total = 0
    capped = []
    tok = {}
    for i, c in zip(kept, counts):
        if total >= budget:
            break
        capped.append(i)
        tok[i] = c
        total += c

    # ---- explicit sequence: [L (+ extra passes)] + [H]; no global shuffle -----
    return _build_order(capped, tok, rng)
