"""Curator: three-tier curriculum (general prose -> near-miss wiki -> pure target-like).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled, so the sequence is built
explicitly to hold between T and 1.02*T tokens.

Tier A: best copy per title/prefix group whose bigram hit rate passes the LOO threshold.
Tier B: near-miss wiki articles (titles not in A), at most 0.35*T tokens.
Tier C: clean general English prose from other domains, at most 0.25*T tokens.

Order: C (shuffled) -> B mixed with extra passes of A (shuffled together)
       -> one shuffled pass of A capped to its highest-scoring 0.6*T tokens.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
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
OVERSHOOT = 1.02         # hard upper bound on sequence length, in units of T
FILL_TARGET = 1.005      # aim slightly above T so the trainer never wraps
TIKTOKEN_DEADLINE = 100  # seconds since start after which we stop exact tokenising

# Tier B
B_SCORE_SLACK = 0.08
B_VOCAB_SLACK = 0.05
B_MAX_FRAC = 0.35

# Tier C
C_SCORE_MULT = 0.6
C_VOCAB_SLACK = 0.10
C_MIN_CLEAN = 0.92
C_MIN_TOKENS = 100
C_MAX_SPECIAL = 0.015
C_TOKENS_PER_SENT = 60
C_MAX_FRAC = 0.25
C_SPECIAL_CHARS = "{}[]<>|=;#_/\\"

VOCAB_PCTL = 2.0
A_LAST_FRAC = 0.6

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}
_NON_LETTERS = re.compile(r"[^a-z]+")


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


def _vocab_frac(toks, contains):
    """Fraction of alphabetic tokens for which contains(token) is true."""
    n = 0
    hits = 0
    for t in toks:
        if t.isalpha():
            n += 1
            if contains(t):
                hits += 1
    return hits / n if n else 0.0


def _title_key(text):
    """Return (is_wiki, key) where key is the lowercased letters of a '= title =' first line."""
    stripped = text.lstrip()
    line = stripped.split("\n", 1)[0].strip()
    if len(line) >= 3 and line[0] == "=" and line[-1] == "=":
        inner = line[1:-1].strip()
        if inner and inner[0] != "=" and inner[-1] != "=":
            return True, _NON_LETTERS.sub("", inner.lower())
    return False, ""


def _special_frac(text):
    if not text:
        return 1.0
    n = 0
    for ch in C_SPECIAL_CHARS:
        n += text.count(ch)
    return n / len(text)


class _TokCounter:
    """Cached gpt2 token counts; falls back to len/4 if tiktoken is unavailable or late."""

    def __init__(self, docs, t0):
        self.docs = docs
        self.t0 = t0
        self.cache = {}
        self.enc = None
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None

    def ensure(self, idxs, chunk=256):
        docs = self.docs
        cache = self.cache
        todo = [i for i in idxs if i not in cache]
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            done = False
            if self.enc is not None and time.time() - self.t0 <= TIKTOKEN_DEADLINE:
                try:
                    batch = self.enc.encode_ordinary_batch([docs[i] for i in part])
                    for i, ids in zip(part, batch):
                        cache[i] = max(1, len(ids))
                    done = True
                except Exception:
                    self.enc = None
            if not done:
                for i in part:
                    cache[i] = max(1, len(docs[i]) // 4)

    def __getitem__(self, i):
        c = self.cache.get(i)
        if c is None:
            self.ensure([i])
            c = self.cache[i]
        return c

    def total(self, idxs):
        self.ensure(idxs)
        cache = self.cache
        return sum(cache[i] for i in idxs)


def _take(idxs, budget, tc, chunk=256, max_misses=20):
    """Take docs in the given order with total tokens <= budget."""
    out = []
    total = 0
    if budget <= 0:
        return out, total
    misses = 0
    for s in range(0, len(idxs), chunk):
        part = idxs[s:s + chunk]
        tc.ensure(part)
        for i in part:
            c = tc[i]
            if total + c <= budget:
                out.append(i)
                total += c
                misses = 0
            else:
                misses += 1
                if misses >= max_misses:
                    return out, total
            if total >= budget:
                return out, total
    return out, total


def _cap_until(idxs, budget, tc):
    """Take docs in order until the running total reaches budget (may overshoot by one doc)."""
    tc.ensure(idxs)
    out = []
    total = 0
    for i in idxs:
        if total >= budget:
            break
        out.append(i)
        total += tc[i]
    return out, total


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- reference bigram set / vocabulary + leave-one-out calibration -------
    df = Counter()            # number of target docs containing each bigram
    wdf = Counter()           # number of target docs containing each word
    target_heads = []
    for t in target_sample:
        toks = t.lower().split()
        df.update(set(zip(toks, toks[1:])))
        wdf.update(set(toks))
        target_heads.append(toks[:SCORE_TOKENS])
    B_set = set(df)
    V_set = set(wdf)

    loo = []
    loo_v = []
    for toks in target_heads:
        if len(toks) < 2:
            continue
        # present in the other 199 docs iff doc-frequency is >= 2
        loo.append(_hit_rate(toks, lambda b: df.get(b, 0) >= 2))
        loo_v.append(_vocab_frac(toks, lambda w: wdf.get(w, 0) >= 2))
    thr = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    thr = max(thr, 0.0)
    v_p = float(np.percentile(loo_v, VOCAB_PCTL)) if loo_v else 0.0
    v_thr_b = v_p - B_VOCAB_SLACK
    v_thr_c = v_p - C_VOCAB_SLACK

    # ---- score every doc that passes the basic junk filter -------------------
    # record: (score, idx, vocab_frac, clean_frac, n_head_tokens, is_wiki, group_key)
    records = []
    b_contains = B_set.__contains__
    v_contains = V_set.__contains__
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        toks = _head_tokens(text)
        ntok = len(toks)
        if ntok < MIN_TOKENS:
            continue
        clean = _clean_frac(text)
        if clean < MIN_CLEAN_FRAC:
            continue
        is_wiki, tkey = _title_key(text)
        if is_wiki and tkey:
            key = "t:" + tkey
        else:
            norm = " ".join(text[:PREFIX_CHARS * 20].lower().split())[:PREFIX_CHARS]
            key = "p:" + norm
        score = _hit_rate(toks, b_contains)
        vfrac = _vocab_frac(toks, v_contains)
        records.append((score, i, vfrac, clean, ntok, is_wiki, key))

    if not records:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    def better(r, cur):
        return cur is None or r[0] > cur[0] or (r[0] == cur[0] and r[1] < cur[1])

    def by_score(recs):
        return [r[1] for r in sorted(recs, key=lambda r: (-r[0], r[1]))]

    tc = _TokCounter(docs, t0)

    # ---- Tier A: best copy per group, score >= thr ---------------------------
    best = {}
    for r in records:
        if better(r, best.get(r[6])):
            best[r[6]] = r
    a_recs = [r for r in best.values() if r[0] >= thr]

    if not a_recs:
        # Nothing passed the threshold: fall back to the best-scoring clean docs.
        fallback, tok_f = _cap_until(by_score(best.values()), T * OVERSHOOT, tc)
        print("curate: A empty, fallback docs=%d tokens/T=%.3f" % (len(fallback), tok_f / T))
        rng.shuffle(fallback)
        return fallback

    a_keys = {r[6] for r in a_recs}
    a_sorted = by_score(a_recs)
    tok_a = tc.total(a_sorted)

    if tok_a >= OVERSHOOT * T:
        kept, tok_k = _cap_until(a_sorted, T * OVERSHOOT, tc)
        print("curate: A=%.3fT (%d docs, capped to %.3fT, %d docs) B=0.000T (0 docs) "
              "C=0.000T (0 docs) extra_A_passes=0"
              % (tok_a / T, len(a_sorted), tok_k / T, len(kept)))
        rng.shuffle(kept)
        return kept

    # ---- Tier B: near-miss wiki articles -------------------------------------
    b_best = {}
    b_floor = thr - B_SCORE_SLACK
    for r in records:
        if not r[5]:
            continue
        if r[6] in a_keys:
            continue
        if r[0] < b_floor or r[2] < v_thr_b:
            continue
        if better(r, b_best.get(r[6])):
            b_best[r[6]] = r
    b_idx, tok_b = _take(by_score(b_best.values()), B_MAX_FRAC * T, tc)
    b_keys = set(b_best)

    # ---- Tier C: clean general English prose ---------------------------------
    c_budget = min(C_MAX_FRAC * T, T - tok_a - tok_b)
    c_idx, tok_c = [], 0
    if c_budget > 0:
        c_best = {}
        c_floor = C_SCORE_MULT * thr
        for r in records:
            score, i, vfrac, clean, ntok, is_wiki, key = r
            if is_wiki:
                continue
            if score < c_floor or vfrac < v_thr_c:
                continue
            if clean < C_MIN_CLEAN or ntok < C_MIN_TOKENS:
                continue
            if key in a_keys or key in b_keys:
                continue
            text = docs[i]
            if _special_frac(text) >= C_MAX_SPECIAL:
                continue
            ntot = len(text.split()) if ntok >= SCORE_TOKENS else ntok
            # '. ' also matches the tokenised form ' . '
            if text.count(". ") * C_TOKENS_PER_SENT < ntot:
                continue
            if better(r, c_best.get(key)):
                c_best[key] = r
        c_idx, tok_c = _take(by_score(c_best.values()), c_budget, tc)

    # ---- Sequence -------------------------------------------------------------
    # First: all of C, shuffled.
    first = list(c_idx)
    rng.shuffle(first)

    # Last: one pass of A capped to its highest-scoring 0.6*T tokens.
    last, tok_last = _cap_until(a_sorted, A_LAST_FRAC * T, tc)

    # Middle: B plus extra passes of A as needed to fill.
    target = FILL_TARGET * T
    hard = OVERSHOOT * T
    total = tok_c + tok_b + tok_last
    middle = list(b_idx)
    extra_passes = 0
    while total < target:
        order = list(a_sorted)
        rng.shuffle(order)
        added = False
        for i in order:
            if total >= target:
                break
            c = tc[i]
            if total + c > hard:
                continue
            middle.append(i)
            total += c
            added = True
        if not added:
            break
        extra_passes += 1
    rng.shuffle(middle)
    rng.shuffle(last)

    print("curate: A=%.3fT (%d docs) B=%.3fT (%d docs) C=%.3fT (%d docs) "
          "extra_A_passes=%d last_A=%.3fT total=%.3fT thr=%.4f elapsed=%.1fs"
          % (tok_a / T, len(a_sorted), tok_b / T, len(b_idx), tok_c / T, len(c_idx),
             extra_passes, tok_last / T, total / T, thr, time.time() - t0))

    return first + middle + last
