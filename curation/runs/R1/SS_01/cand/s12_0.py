"""Curator: two-tier target-likeness selection with an explicit training sequence.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  * score every full document in 250-token windows against the 200-doc reference
    (word-bigram hit rate: overall mean and worst window; unigram vocab coverage)
  * calibrate m5 / w1 / v2 by leave-one-out on the target sample
  * best-copy dedupe
  * tier A: mean >= m5-0.02 AND minw >= w1-0.05 AND vocab >= v2-0.03
  * tier B: '= title =' docs not in A, mean >= m5-0.10, minw >= w1-0.20,
            vocab >= v2-0.05, title not already in A, at most
            min(T - tokens(A), 0.20*T) tokens, highest mean first
  * sequence: [early: B mixed with extra A passes] + [final: one pass of the
    highest-mean 0.5*T tokens of A]; total between T and 1.02*T tokens.
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

# ---- scoring / calibration ---------------------------------------------------
WINDOW = 250             # word tokens per window
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85

A_MEAN_SLACK, A_MINW_SLACK, A_VOCAB_SLACK = 0.02, 0.05, 0.03
B_MEAN_SLACK, B_MINW_SLACK, B_VOCAB_SLACK = 0.10, 0.20, 0.05

# ---- budget / sequence -------------------------------------------------------
B_MAX_FRAC = 0.20        # B takes at most this share of T
FINAL_FRAC = 0.50        # final block: highest-mean A tokens
TOTAL_TARGET = 1.01      # aim for the middle of [T, 1.02*T]
A_FULL_FRAC = 1.02       # A alone is enough above this
A_CAP_TARGET = 1.005     # cap used when A alone is enough
A_MIN_FRAC = 0.30        # below this, fall back to the legacy pipeline

TIKTOKEN_DEADLINE = 100  # seconds since start after which we stop exact tokenising

# ---- legacy pipeline constants -----------------------------------------------
SCORE_TOKENS = 2000
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}

_TITLE_RE = re.compile(r"\s*=[ \t]+([^=\n][^\n]*?)[ \t]+=[ \t]*(?:\r?\n|$)")


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
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


def _title_key(text):
    """Lowercased top-level '= title =' heading at the start of the doc, or None."""
    m = _TITLE_RE.match(text)
    if not m:
        return None
    key = " ".join(m.group(1).lower().split())
    return key or None


def _score(toks, b_contains, v_contains):
    """(mean bigram hit rate, worst 250-token window hit rate, vocab coverage)."""
    n_tok = len(toks)
    if n_tok < 2:
        return 0.0, 0.0, 0.0
    hits = list(map(b_contains, zip(toks, toks[1:])))
    n = len(hits)
    mean = sum(hits) / n
    if n <= WINDOW:
        minw = mean
    else:
        minw = 1.0
        half = WINDOW // 2
        for s in range(0, n, WINDOW):
            seg = hits[s:s + WINDOW]
            if len(seg) < half:
                break  # short tail: already reflected in the mean
            r = sum(seg) / len(seg)
            if r < minw:
                minw = r
    vocab = sum(map(v_contains, toks)) / n_tok
    return mean, minw, vocab


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


# ------------------------------------------------------------------------------
# legacy pipeline (fallback)
# ------------------------------------------------------------------------------
def _legacy_select(docs, target_sample, seed, t0):
    rng = random.Random(seed)

    df = Counter()
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
        loo.append(_hit_rate(toks, lambda b: df.get(b, 0) >= 2))
    threshold = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    threshold = max(threshold, 0.0)

    seen_full = set()
    seen_prefix = set()
    cand = []
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

    cand.sort(key=lambda x: (-x[0], x[1]))
    kept = [i for s, i in cand if s >= threshold]

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
        budget = T * OVERSHOOT
        total = 0
        for s, i in cand:
            if total >= budget:
                break
            kept.append(i)
            total += max(1, len(docs[i]) // 4)

    rng.shuffle(kept)
    return kept


# ------------------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------------------
def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- reference sets + leave-one-out calibration ---------------------------
    bdf = Counter()           # doc frequency of each bigram in the target sample
    wdf = Counter()           # doc frequency of each word in the target sample
    target_toks = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        toks = t.lower().split()
        if len(toks) < 2:
            continue
        bdf.update(set(zip(toks, toks[1:])))
        wdf.update(set(toks))
        target_toks.append(toks)

    if len(target_toks) < 5:
        print("[curate] too few usable target docs -> legacy pipeline")
        return _legacy_select(docs, target_sample, seed, t0)

    # a bigram/word is in the other 199 docs iff its doc frequency is >= 2
    b_loo = {b for b, c in bdf.items() if c >= 2}
    w_loo = {w for w, c in wdf.items() if c >= 2}
    loo = [_score(toks, b_loo.__contains__, w_loo.__contains__) for toks in target_toks]
    m5 = float(np.percentile([s[0] for s in loo], 5))
    w1 = float(np.percentile([s[1] for s in loo], 1))
    v2 = float(np.percentile([s[2] for s in loo], 2))
    del b_loo, w_loo, loo, target_toks

    B_set = set(bdf)
    V_set = set(wdf)
    del bdf, wdf
    b_contains = B_set.__contains__
    v_contains = V_set.__contains__

    # ---- score full documents, junk filter, best-copy dedupe ------------------
    best = {}                 # prefix key -> (quality, idx, mean, minw, vocab)
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        toks = text.lower().split()
        if len(toks) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue
        mean, minw, vocab = _score(toks, b_contains, v_contains)
        key = hash(" ".join(toks[:80])[:PREFIX_CHARS])
        quality = (round(mean, 2), round(minw, 1), len(toks), -i)
        cur = best.get(key)
        if cur is None or quality > cur[0]:
            best[key] = (quality, i, mean, minw, vocab)
    del B_set, V_set

    survivors = sorted((v[1], v[2], v[3], v[4]) for v in best.values())
    del best

    # ---- tiers ------------------------------------------------------------------
    a_mean, a_minw, a_vocab = m5 - A_MEAN_SLACK, w1 - A_MINW_SLACK, v2 - A_VOCAB_SLACK
    b_mean, b_minw, b_vocab = m5 - B_MEAN_SLACK, w1 - B_MINW_SLACK, v2 - B_VOCAB_SLACK

    mean_of = {}
    A = []
    b_cand = []
    for i, mean, minw, vocab in survivors:
        if mean >= a_mean and minw >= a_minw and vocab >= a_vocab:
            A.append(i)
            mean_of[i] = mean
        elif mean >= b_mean and minw >= b_minw and vocab >= b_vocab:
            b_cand.append(i)
            mean_of[i] = mean
    del survivors

    if not A:
        print("[curate] tier A empty -> legacy pipeline")
        return _legacy_select(docs, target_sample, seed, t0)

    # highest mean first, deterministic tie-break on index
    A.sort(key=lambda i: (-mean_of[i], i))
    tok = dict(zip(A, _count_tokens([docs[i] for i in A], t0)))
    tokA = sum(tok[i] for i in A)

    if tokA < A_MIN_FRAC * T:
        print("[curate] tokens(A)/T=%.3f docs(A)=%d < %.2f -> legacy pipeline"
              % (tokA / T, len(A), A_MIN_FRAC))
        return _legacy_select(docs, target_sample, seed, t0)

    # ---- A alone is enough ------------------------------------------------------
    if tokA >= A_FULL_FRAC * T:
        out = []
        total = 0
        for i in A:
            if total >= A_CAP_TARGET * T:
                break
            out.append(i)
            total += tok[i]
        print("[curate] tokens(A)/T=%.3f (capped to %.3f) tokens(B)/T=0.000 "
              "docs(A)=%d (kept %d) docs(B)=0 extra_A_passes=0.00"
              % (tokA / T, total / T, len(A), len(out)))
        rng.shuffle(out)
        return out

    # ---- tier B -------------------------------------------------------------------
    B = []
    tokB = 0
    b_budget = min(T - tokA, B_MAX_FRAC * T)
    if b_budget > 0 and b_cand:
        seen_titles = set()
        for i in A:
            k = _title_key(docs[i])
            if k is not None:
                seen_titles.add(k)

        b_cand.sort(key=lambda i: (-mean_of[i], i))
        pool_b = []
        est = 0
        for i in b_cand:
            if est >= 2 * b_budget:
                break
            k = _title_key(docs[i])
            if k is None or k in seen_titles:
                continue
            seen_titles.add(k)
            pool_b.append(i)
            est += max(1, len(docs[i]) // 4)

        if pool_b:
            cb = _count_tokens([docs[i] for i in pool_b], t0)
            for i, c in zip(pool_b, cb):
                if tokB + c > b_budget:
                    continue
                B.append(i)
                tok[i] = c
                tokB += c

    # ---- final block: highest-mean 0.5*T tokens of A ----------------------------
    final = []
    tokF = 0
    for i in A:
        if tokF >= FINAL_FRAC * T:
            break
        final.append(i)
        tokF += tok[i]
    final_set = set(final)
    rest = [i for i in A if i not in final_set]   # must appear in the early block
    tokR = tokA - tokF

    # ---- early block: B + extra A material ----------------------------------------
    need = TOTAL_TARGET * T - tokF - tokB          # extra A tokens wanted
    if need < 0:
        need = 0
    n_full = int(need // tokA)
    rem = need - n_full * tokA

    frac = []
    tok_frac = 0
    if n_full == 0:
        # no whole pass: docs outside the final cap are mandatory, then the
        # highest-mean A docs fill the remainder
        frac = list(rest)
        tok_frac = tokR
        for i in A:
            if tok_frac >= rem:
                break
            if i not in final_set:
                continue
            frac.append(i)
            tok_frac += tok[i]
    elif rem > 0:
        for i in A:
            if tok_frac >= rem:
                break
            frac.append(i)
            tok_frac += tok[i]

    seq_a = []
    rng.shuffle(frac)
    seq_a.extend(frac)
    for _ in range(n_full):
        p = list(A)
        rng.shuffle(p)
        seq_a.extend(p)

    b_shuf = list(B)
    rng.shuffle(b_shuf)

    # mix B into the A material at seeded-random positions, keeping pass order
    labels = [0] * len(seq_a) + [1] * len(b_shuf)
    rng.shuffle(labels)
    it_a = iter(seq_a)
    it_b = iter(b_shuf)
    early = [next(it_b) if lab else next(it_a) for lab in labels]

    final_pass = list(final)
    rng.shuffle(final_pass)

    out = early + final_pass
    total = n_full * tokA + tok_frac + tokB + tokF

    print("[curate] tokens(A)/T=%.3f tokens(B)/T=%.3f docs(A)=%d docs(B)=%d "
          "extra_A_passes=%.2f (whole=%d) final/T=%.3f total/T=%.3f "
          "m5=%.3f w1=%.3f v2=%.3f elapsed=%.1fs"
          % (tokA / T, tokB / T, len(A), len(B),
             n_full + tok_frac / tokA, n_full, tokF / T, total / T,
             m5, w1, v2, time.time() - t0))
    return out
