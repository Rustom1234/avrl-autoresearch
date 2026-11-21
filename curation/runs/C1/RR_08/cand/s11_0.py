"""Curator: greedy selection of pool documents by lexical overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Use tiktoken on a seeded sample of the pool to estimate tokens per character,
   so the token budget T can be applied without tokenizing the whole pool.
2. Windows: each document is represented by up to three 3000-char windows
   (start, middle, end), so the whole document is checked, not only its head.
3. Damaged-text filter (experiment-3 word-order filter, per window): the fraction
   of adjacent token bigrams that are "known" (count >= 3 in target windows plus
   a seeded pool sample). A doc's hit is the minimum over its windows. Docs below
   0.6 * (5th percentile of the leave-one-out target hit) go to the fallback list.
4. Unigram LLR word weights (built from first-4000-char counts, unchanged); a
   doc's score is the mean weight over the pooled words of its windows. The junk
   checks (type/token ratio, alphabetic fraction) are applied to the pooled windows.
5. Greedy fill by score to 1.08 T. The docs filling the first 1.00 T (best scores)
   are shuffled and placed first; the marginal remainder is shuffled and appended,
   so the trainer's cut-off drops only the lowest-scoring documents.

NOTE (unverified): the experiment log records experiments 3 and 10 as rejected
although both scored below the best. That record could not be checked from
inside this file. If the filter was rejected for a reason other than score
(e.g. run time), set USE_BIGRAM_FILTER = False: that drops the filter and keeps
the windows, the pooled scoring and the ordering. A wall-clock guard
(TIME_GUARD_S) also turns the filter off and falls back to the first window
only if selection is running long.
"""
import os
import re
import math
import time
import random
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_BTOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")
_WS = re.compile(r"\s+")

SCORE_CHARS = 4000      # characters of each doc used for the word-weight counts
WIN = 3000              # window size in characters
TOKEN_SAMPLE = 600      # docs tokenized to estimate token statistics
BG_SAMPLE = 4000        # docs used for the background distributions
LAMBDA = 0.7            # weight of target distribution in the mixture
MAX_W = 4.0             # cap on a single word's weight
BUDGET_MARGIN = 1.08    # select a bit more than T to absorb estimation error

USE_BIGRAM_FILTER = True
BIGRAM_MIN_COUNT = 3
THR_FACTOR = 0.6
THR_PERCENTILE = 5.0
TIME_GUARD_S = 120.0    # after this, skip the filter and use only the first window


def _words(text):
    return _WORD.findall(text[:SCORE_CHARS].lower())


def _windows(d):
    n = len(d)
    if n <= WIN:
        return [d]
    if n <= 2 * WIN:
        return [d[:WIN], d[-WIN:]]
    m = n // 2
    h = WIN // 2
    return [d[:WIN], d[m - h:m - h + WIN], d[-WIN:]]


def _percentile(vals, q):
    v = sorted(vals)
    if not v:
        return 0.0
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    frac = pos - lo
    return v[lo] * (1.0 - frac) + v[hi] * frac


def _token_stats(docs, rng):
    """Return (average tokens per doc, tokens per character) from a sample."""
    n = len(docs)
    sample = rng.sample(range(n), min(TOKEN_SAMPLE, n))
    tpc = 0.25
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tok = 0
        chars = 0
        for i in sample:
            d = docs[i]
            if not d:
                continue
            tok += len(enc.encode(d, disallowed_special=()))
            chars += len(d)
        if chars > 0 and tok > 0:
            tpc = tok / chars
    except Exception:
        pass
    total_chars = sum(len(d) for d in docs)
    avg_tokens = tpc * total_chars / max(1, n)
    return avg_tokens, tpc


def select(docs, target_sample):
    t0 = time.monotonic()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # 1. Token statistics
    avg_tokens, tpc = _token_stats(docs, rng)
    avg_tokens = max(avg_tokens, 1.0)

    # 2. Word frequencies (first-4000-char counts, unchanged)
    tgt = Counter()
    for t in target_sample:
        tgt.update(_words(t))
    n_t = sum(tgt.values())

    bg_idx = rng.sample(range(n), min(BG_SAMPLE, n))
    bg = Counter()
    for i in bg_idx:
        bg.update(_words(docs[i]))
    n_b = sum(bg.values())

    if n_t == 0 or n_b == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    floor_b = 0.5 / n_b
    unk_w = math.log(1.0 - LAMBDA)   # word seen in neither target nor background
    weights = {}
    for w, c in bg.items():
        pb = c / n_b
        pt = tgt.get(w, 0) / n_t
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt.items():
        if w not in weights:
            pt = c / n_t
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    # 3. Bigram model for the damaged-text filter
    use_filter = USE_BIGRAM_FILTER
    good = set()
    thr = 0.0
    if use_filter:
        big = Counter()
        tgt_win_bigrams = []   # per target doc: list of bigram lists (one per window)
        for t in target_sample:
            per = []
            if t:
                for w in _windows(t):
                    toks = _BTOK.findall(w.lower())
                    if len(toks) >= 2:
                        bl = list(zip(toks, toks[1:]))
                        big.update(bl)
                        per.append(bl)
            tgt_win_bigrams.append(per)
        for i in bg_idx:
            d = docs[i]
            if not d:
                continue
            for w in _windows(d):
                toks = _BTOK.findall(w.lower())
                if len(toks) >= 2:
                    big.update(zip(toks, toks[1:]))

        # leave-one-out calibration on the target docs
        loo = []
        bget = big.get
        for per in tgt_win_bigrams:
            if not per:
                continue
            own = Counter()
            for bl in per:
                own.update(bl)
            oget = own.get
            mh = 1.0
            for bl in per:
                h = 0
                for b in bl:
                    if bget(b, 0) - oget(b, 0) >= BIGRAM_MIN_COUNT:
                        h += 1
                h = h / len(bl)
                if h < mh:
                    mh = h
            loo.append(mh)
        if loo:
            thr = THR_FACTOR * _percentile(loo, THR_PERCENTILE)
            good = {b for b, c in big.items() if c >= BIGRAM_MIN_COUNT}
        else:
            use_filter = False
        del big, tgt_win_bigrams
    good_has = good.__contains__

    # 4. Score documents, drop duplicates, junk and damaged text
    wget = weights.get
    wfind = _WORD.findall
    bfind = _BTOK.findall
    seen = set()
    scored = []
    rest = []   # low-priority fallback (duplicates excluded)
    full = True  # False once the time guard has tripped
    for k, d in enumerate(docs):
        if not d:
            continue
        key = hash(_WS.sub(" ", d.strip().lower()))
        if key in seen:
            continue
        seen.add(key)

        if full and (k & 255) == 0 and time.monotonic() - t0 > TIME_GUARD_S:
            full = False

        wins = _windows(d) if full else [d[:WIN]]
        lows = [w.lower() for w in wins]

        ws = []
        chars = 0
        for lw in lows:
            ws.extend(wfind(lw))
            chars += len(lw)
        nw = len(ws)
        if nw < 20:
            rest.append(k)
            continue
        # junk checks on the pooled windows
        if nw >= 100 and len(set(ws)) / nw < 0.12:
            rest.append(k)
            continue
        if sum(map(len, ws)) / max(1, chars) < 0.5:
            rest.append(k)
            continue

        if use_filter and full:
            mh = 1.0
            for lw in lows:
                toks = bfind(lw)
                nb = len(toks) - 1
                if nb < 1:
                    continue
                h = sum(map(good_has, zip(toks, toks[1:]))) / nb
                if h < mh:
                    mh = h
            if mh < thr:
                rest.append(k)
                continue

        s = 0.0
        for w in ws:
            s += wget(w, unk_w)
        scored.append((s / nw, k))

    scored.sort(key=lambda x: -x[0])

    # 5. Greedy fill of the token budget, best documents first
    budget = T * BUDGET_MARGIN
    chosen = []
    used = 0.0
    for s, i in scored:
        if used >= budget:
            break
        chosen.append(i)
        used += max(1.0, len(docs[i]) * tpc)

    # Fallback: only if the surviving set is far too small
    if used < 0.25 * T:
        have = set(chosen)
        for i in rest:
            if used >= budget:
                break
            if i not in have:
                chosen.append(i)
                used += max(1.0, len(docs[i]) * tpc)

    if not chosen:
        chosen = list(range(n))
        rng.shuffle(chosen)
        return chosen

    # 6. Ordering: best docs filling the first 1.00 T shuffled first, marginal part last
    head = []
    tail = []
    cum = 0.0
    for i in chosen:
        if cum < T:
            head.append(i)
        else:
            tail.append(i)
        cum += max(1.0, len(docs[i]) * tpc)

    rng.shuffle(head)
    rng.shuffle(tail)
    return head + tail
