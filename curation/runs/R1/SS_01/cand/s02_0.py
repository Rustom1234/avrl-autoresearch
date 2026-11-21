"""Curator: whole-document damage detection + best-copy dedupe + target-likeness filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
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

WINDOW = 250             # word tokens per scoring window
MIN_LAST_WINDOW = 100    # a final window shorter than this is ignored for the min
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0               # mean window hit rate percentile
MARGIN = 0.02
MIN_PCTL = 1.0           # min window hit rate percentile
MIN_MARGIN = 0.05
VOCAB_PCTL = 2.0         # in-vocabulary fraction percentile
VOCAB_MARGIN = 0.03
SHINGLE_LEN = 12
SHINGLE_START = 20
SHINGLE_END = 80
N_SHINGLES = 3
OVERSHOOT = 1.02         # keep slightly more than T so the cap never causes cycling
TIKTOKEN_DEADLINE = 100  # seconds since start after which we stop exact tokenising

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}

_TITLE_RE = re.compile(r"^=\s+([^=].*?)\s+=$")
_NON_LETTERS = re.compile(r"[^a-z]")


def _clean_frac(text):
    if not text:
        return 0.0
    bad = len(text.translate(_DELETE_ALLOWED))
    return 1.0 - bad / len(text)


def _window_stats(toks, contains):
    """(mean window hit rate, min window hit rate) over consecutive windows."""
    n = len(toks)
    if n < 2:
        return 0.0, 0.0
    flags = list(map(contains, zip(toks, toks[1:])))
    rates = []
    full = []
    for s in range(0, n, WINDOW):
        e = min(s + WINDOW, n)
        m = e - s - 1            # bigrams fully inside this window
        if m <= 0:
            continue
        r = sum(flags[s:e - 1]) / m
        rates.append(r)
        if e - s >= MIN_LAST_WINDOW:
            full.append(r)
    if not rates:
        return 0.0, 0.0
    mean = sum(rates) / len(rates)
    mn = min(full) if full else min(rates)
    return mean, mn


def _vocab_frac(toks, contains):
    alpha = [t for t in toks if t.isalpha()]
    if not alpha:
        return 0.0
    return sum(map(contains, alpha)) / len(alpha)


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


def _dedupe_keys(text, toks):
    """Grouping keys for one doc: a title/prefix key plus body shingle keys."""
    keys = []
    stripped = text.strip()
    first_line = stripped.split("\n", 1)[0].strip()
    m = _TITLE_RE.match(first_line)
    title_key = ""
    body = toks
    if m:
        title_key = _NON_LETTERS.sub("", m.group(1).lower())
        body = toks[len(first_line.split()):]
    if title_key:
        keys.append(("t", title_key))
    else:
        prefix = " ".join(toks[:PREFIX_CHARS])[:PREFIX_CHARS]
        keys.append(("p", hash(prefix)))

    region = body[SHINGLE_START:SHINGLE_END]
    for k in range(N_SHINGLES):
        sh = region[k * SHINGLE_LEN:(k + 1) * SHINGLE_LEN]
        if len(sh) < SHINGLE_LEN:
            break
        keys.append(("s", hash(" ".join(sh))))
    return keys


def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- reference sets + leave-one-out calibration --------------------------
    df = Counter()            # number of target docs containing each bigram
    wdf = Counter()           # number of target docs containing each word
    target_toks = []
    for t in target_sample:
        toks = t.lower().split()
        df.update(set(zip(toks, toks[1:])))
        wdf.update(set(toks))
        target_toks.append(toks)
    B = set(df)
    V = set(wdf)
    # an item is in the other 199 docs iff its doc-frequency is >= 2
    B2 = {b for b, c in df.items() if c >= 2}
    V2 = {w for w, c in wdf.items() if c >= 2}

    loo_mean, loo_min, loo_vocab = [], [], []
    for toks in target_toks:
        if len(toks) < 2:
            continue
        mean, mn = _window_stats(toks, B2.__contains__)
        loo_mean.append(mean)
        loo_min.append(mn)
        loo_vocab.append(_vocab_frac(toks, V2.__contains__))

    if loo_mean:
        mean_thr = max(float(np.percentile(loo_mean, PCTL)) - MARGIN, 0.0)
        min_thr = max(float(np.percentile(loo_min, MIN_PCTL)) - MIN_MARGIN, 0.0)
        vocab_thr = max(float(np.percentile(loo_vocab, VOCAB_PCTL)) - VOCAB_MARGIN, 0.0)
    else:
        mean_thr = min_thr = vocab_thr = 0.0

    # ---- score every non-junk doc, build dedupe groups ------------------------
    b_contains = B.__contains__
    v_contains = V.__contains__
    idxs = []       # doc index per candidate
    means = []      # mean window hit rate
    passed = []     # passes all calibrated thresholds
    parent = []     # union-find over candidates
    key_owner = {}

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        toks = text.lower().split()
        if len(toks) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue

        mean, mn = _window_stats(toks, b_contains)
        ok = mean >= mean_thr and mn >= min_thr
        if ok:
            ok = _vocab_frac(toks, v_contains) >= vocab_thr

        pos = len(idxs)
        idxs.append(i)
        means.append(mean)
        passed.append(ok)
        parent.append(pos)

        for key in _dedupe_keys(text, toks):
            other = key_owner.get(key)
            if other is None:
                key_owner[key] = pos
            else:
                ra = _find(parent, other)
                rb = _find(parent, pos)
                if ra != rb:
                    if ra < rb:
                        parent[rb] = ra
                    else:
                        parent[ra] = rb

    if not idxs:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # ---- best copy per group ---------------------------------------------------
    def best_per_group(positions):
        best = {}
        for pos in positions:
            root = _find(parent, pos)
            i = idxs[pos]
            rank = (-means[pos], -len(docs[i]), i)
            cur = best.get(root)
            if cur is None or rank < cur[0]:
                best[root] = (rank, pos)
        return [(means[pos], idxs[pos]) for _, pos in best.values()]

    all_pos = range(len(idxs))
    kept_scored = best_per_group([p for p in all_pos if passed[p]])
    fallback = not kept_scored
    if fallback:
        # Nothing passed the thresholds: fall back to the best-scoring clean docs.
        kept_scored = best_per_group(all_pos)

    # best first, deterministic tie-break on index
    kept_scored.sort(key=lambda x: (-x[0], x[1]))
    kept = [i for _, i in kept_scored]

    # ---- cap at ~T tokens, keeping highest-scoring -----------------------------
    if fallback:
        counts = [max(1, len(docs[i]) // 4) for i in kept]
    else:
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

    # ---- seeded shuffle ----------------------------------------------------------
    rng.shuffle(kept)
    return kept
