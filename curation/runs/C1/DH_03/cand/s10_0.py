"""Curator: experiment-8 seed selection plus one self-training round.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stage A: junk guards, top-5000 target bigram score, held-out threshold
         (2-means valley threshold for article-format docs), dedup -> seed set S.
Stage B: top-100,000 bigrams of S + target_sample, rescore the whole pool,
         threshold calibrated on the target sample, dedup, exact token counting,
         repeated shuffled passes.

Assumption: the experiment 8 source was not available when this was written.
The 2-means valley is implemented as the midpoint of the two 1-D k-means
centroids of the article-format docs' stage-A scores. It replaces the held-out
threshold for those docs. Stage A has no 0.5 T fill.
"""
import os
import random
import re
import time
import hashlib
from collections import Counter

import numpy as np
import tiktoken

from common import T

TOP_BIGRAMS = 5000
TOP_BIGRAMS_2 = 100000
MIN_CHARS = 200
PREFIX_CHARS = 300
TARGET_PCT_A = 5
TARGET_PCT_B = 3
THR_SLACK = 0.95
SEED_SLACK = 0.9
MIN_FILL_FRACTION = 0.08
WORD_CAP = 3000
STAGE_A_TIME_LIMIT = 50.0  # seconds; beyond this, stage B uses WORD_CAP
TOKEN_BATCH = 2000

_ODD_RE = re.compile(r"[^A-Za-z0-9\s.,;:!?'\"()\-]")
_HEADER_RE = re.compile(r"^= [^=].* =$")


def _words(text, cap=None):
    w = text.split()
    if cap is not None and len(w) > cap:
        w = w[:cap]
    return w


def _bigram_set(texts, k, cap=None):
    counts = Counter()
    for t in texts:
        w = _words(t, cap)
        counts.update(zip(w, w[1:]))
    return {b for b, _ in counts.most_common(k)}


def _score(text, bigrams, cap=None):
    w = _words(text, cap)
    n = len(w) - 1
    if n <= 0:
        return 0.0
    return sum(map(bigrams.__contains__, zip(w, w[1:]))) / n


def _nonascii_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (n - len(text.encode("ascii", "ignore"))) / n


def _odd_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return len(_ODD_RE.findall(text)) / n


def _hashes(text):
    norm = " ".join(text.lower().split())
    full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
    prefix = hashlib.md5(norm[:PREFIX_CHARS].encode("utf-8", "ignore")).digest()
    return full, prefix


def _is_article(text):
    first = text.lstrip("\n").split("\n", 1)[0].strip()
    return bool(_HEADER_RE.match(first))


def _two_means_valley(values):
    """Midpoint between the two centroids of a 1-D 2-means; None if degenerate."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 10:
        return None
    lo, hi = np.percentile(x, 10), np.percentile(x, 90)
    if hi - lo < 1e-9:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        upper = x >= mid
        if upper.all() or (~upper).all():
            return None
        new_lo, new_hi = x[~upper].mean(), x[upper].mean()
        if abs(new_lo - lo) < 1e-12 and abs(new_hi - hi) < 1e-12:
            break
        lo, hi = new_lo, new_hi
    return float((lo + hi) / 2.0)


def _token_counts(enc, texts):
    out = []
    for s in range(0, len(texts), TOKEN_BATCH):
        batch = texts[s:s + TOKEN_BATCH]
        out.extend(len(t) for t in enc.encode_ordinary_batch(batch))
    return out


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    target = list(target_sample)
    n = len(docs)

    # ------------------------------------------------------------------
    # Stage A: experiment-8 seed set S.
    # ------------------------------------------------------------------
    half = len(target) // 2
    part_a, part_b = target[:half], target[half:]
    set_a = _bigram_set(part_a, TOP_BIGRAMS)
    set_b = _bigram_set(part_b, TOP_BIGRAMS)
    heldout = [_score(t, set_a) for t in part_b] + [_score(t, set_b) for t in part_a]
    threshold = float(np.percentile(heldout, TARGET_PCT_A)) * THR_SLACK if heldout else 0.0

    final_set = _bigram_set(target, TOP_BIGRAMS)

    max_nonascii = max((_nonascii_share(t) for t in target), default=0.0) * 1.5
    max_odd = max((_odd_share(t) for t in target), default=0.0) * 1.5

    scores = np.zeros(n, dtype=np.float64)
    passed_guard = np.zeros(n, dtype=bool)
    is_article = np.zeros(n, dtype=bool)

    for i, d in enumerate(docs):
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_share(d) > max_nonascii:
            continue
        if _odd_share(d) > max_odd:
            continue
        passed_guard[i] = True
        scores[i] = _score(d, final_set)
        is_article[i] = _is_article(d)

    guard_idx = [i for i in range(n) if passed_guard[i]]
    if not guard_idx:
        return list(range(n))

    art_scores = [scores[i] for i in guard_idx if is_article[i]]
    valley = _two_means_valley(art_scores)
    art_threshold = valley if valley is not None else threshold

    hash_cache = {}

    def get_hashes(i):
        h = hash_cache.get(i)
        if h is None:
            h = _hashes(docs[i])
            hash_cache[i] = h
        return h

    cand_a = [
        i for i in guard_idx
        if scores[i] >= (art_threshold if is_article[i] else threshold)
    ]
    cand_a.sort(key=lambda i: (-scores[i], i))
    seen_full, seen_prefix = set(), set()
    S = []
    for i in cand_a:
        full, prefix = get_hashes(i)
        if full in seen_full or prefix in seen_prefix:
            continue
        seen_full.add(full)
        seen_prefix.add(prefix)
        S.append(i)
    in_S = np.zeros(n, dtype=bool)
    if S:
        in_S[S] = True

    # ------------------------------------------------------------------
    # Stage B: bigger fluency model from S + target, rescore everything.
    # ------------------------------------------------------------------
    cap = WORD_CAP if (time.time() - t0) > STAGE_A_TIME_LIMIT else None

    counts = Counter()
    for i in S:
        w = _words(docs[i], cap)
        counts.update(zip(w, w[1:]))
    for t in target:
        w = _words(t, cap)
        counts.update(zip(w, w[1:]))
    B2 = {b for b, _ in counts.most_common(TOP_BIGRAMS_2)}
    del counts

    scores2 = np.zeros(n, dtype=np.float64)
    for i in guard_idx:
        scores2[i] = _score(docs[i], B2, cap)
    target2 = [_score(t, B2, cap) for t in target]

    # Calibrate on the target sample.
    thr2 = float(np.percentile(target2, TARGET_PCT_B)) * THR_SLACK if target2 else 0.0
    seed_thr = SEED_SLACK * thr2

    # Final selection.
    cands = [
        i for i in guard_idx
        if scores2[i] >= thr2 or (in_S[i] and scores2[i] >= seed_thr)
    ]
    cands.sort(key=lambda i: (-scores2[i], i))

    seen_full, seen_prefix = set(), set()
    kept = []
    for i in cands:
        full, prefix = get_hashes(i)
        if full in seen_full or prefix in seen_prefix:
            continue
        seen_full.add(full)
        seen_prefix.add(prefix)
        kept.append(i)

    # ------------------------------------------------------------------
    # Output: exact token counts, minimal fill, repeated shuffled passes.
    # ------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")
    tok = {}
    for i, c in zip(kept, _token_counts(enc, [docs[i] for i in kept])):
        tok[i] = c
    kept_tokens = sum(tok[i] for i in kept)

    min_tokens = MIN_FILL_FRACTION * T
    if kept_tokens < min_tokens:
        kept_set = set(kept)
        rest = [i for i in guard_idx if i not in kept_set]
        rest.sort(key=lambda i: (-scores2[i], i))
        pos = 0
        while kept_tokens < min_tokens and pos < len(rest):
            batch = []
            while pos < len(rest) and len(batch) < 256:
                i = rest[pos]
                pos += 1
                full, prefix = get_hashes(i)
                if full in seen_full or prefix in seen_prefix:
                    continue
                seen_full.add(full)
                seen_prefix.add(prefix)
                batch.append(i)
            if not batch:
                continue
            for i, c in zip(batch, _token_counts(enc, [docs[i] for i in batch])):
                if kept_tokens >= min_tokens:
                    break
                tok[i] = c
                kept.append(i)
                kept_tokens += c

    if not kept:
        kept = list(guard_idx)
        random.Random(seed).shuffle(kept)
        return kept

    rng = random.Random(seed)
    if kept_tokens <= 0:
        rng.shuffle(kept)
        return kept

    order = []
    total = 0
    goal = T + kept_tokens  # T plus one pass of margin
    while total < goal:
        p = list(kept)
        rng.shuffle(p)
        order.extend(p)
        total += kept_tokens
    return order
