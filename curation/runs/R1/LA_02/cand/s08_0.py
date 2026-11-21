"""Curator: dedupe, drop obvious junk, rank by target-vs-pool unigram LLR, fill to T tokens.
    Improved selection: Core docs (WikiText style) appear twice before filler docs.
    Block-stratified ordering: Group scores into buckets, shuffle blocks, concatenate.
"""
import hashlib
import math
import os
import random
import re
import time
import zlib
from collections import Counter
from itertools import repeat

from common import T

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.25
MIN_ALPHA_SPACE = 0.60
ORDER_TOKENS = 600            # leading whitespace tokens used by the order test
ORDER_REF_MIN_COUNT = 3       # a bigram must occur this often in the target to be in R
ORDER_SHUFFLES = 2            # shuffles averaged for the baseline hit rate
ORDER_EPS = 0.005             # additive smoothing in the ratio
ORDER_RATIO_THRESHOLD = 2.0   # drop if real/shuffled ratio is below this
ORDER_GUARD_FRAC = 0.05       # max tolerated fraction of target docs below threshold
ORDER_HREAL_REL = 0.3         # drop if h_real < this * median target h_real
SCORE_TOKENS = 1000
SCORE_CHARS = 16000
HEADER_BONUS = 1.0
POOL_ALPHA = 0.5          # additive smoothing for the pool unigram model
TARGET_MU_FRAC = 0.05     # target model is smoothed towards the pool (Dirichlet prior)
TOKENISE_CHUNK = 1000
TIME_BUDGET = 150.0       # seconds; after this, token counts are estimated from chars

_HEADER_RE = re.compile(r"^\s*= [^=\n]+ =\s*\n")

def _hit_fraction(toks, ref):
    """Fraction of adjacent bigrams of `toks` that are in `ref`."""
    n = len(toks) - 1
    if n < 1:
        return 0.0
    return sum(map(ref.__contains__, zip(toks, toks[1:]))) / n

def _shuffled_hit_fraction(toks, ref, rng):
    """Mean hit fraction over ORDER_SHUFFLES full shuffles of the same tokens."""
    n = len(toks) - 1
    if n < 1:
        return 0.0
    work = list(toks)
    contains = ref.__contains__
    acc = 0
    for _ in range(ORDER_SHUFFLES):
        rng.shuffle(work)
        acc += sum(map(contains, zip(work, work[1:])))
    return acc / (ORDER_SHUFFLES * n)

def _order_ratio(h_real, h_shuf):
    return (h_real + ORDER_EPS) / (h_shuf + ORDER_EPS)

def _median(values):
    if not values:
        return 0.0
    v = sorted(values)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])

def _percentile(values, q):
    """Linear-interpolated percentile, q in [0, 1]."""
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    frac = pos - lo
    return v[lo] * (1.0 - frac) + v[hi] * frac

def _alpha_space_fraction(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (sum(map(str.isalpha, text)) + sum(map(str.isspace, text))) / n

def _zlib_ratio(text):
    raw = text.encode("utf-8", "replace")
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0:
        return []

    # ---- target statistics --------------------------------------------------
    target_counts = Counter()
    bigram_counts = Counter()
    for text in target_sample:
        toks = text.lower().split()
        target_counts.update(toks)
        bigram_counts.update(zip(toks, toks[1:]))
    ref_bigrams = {bg for bg, c in bigram_counts.items() if c >= ORDER_REF_MIN_COUNT}
    del bigram_counts

    # ---- calibrate the order test on the target sample -----------------------
    target_h_real = []
    target_ratios = []
    if ref_bigrams:
        for j, text in enumerate(target_sample):
            toks = text[:SCORE_CHARS].lower().split()[:ORDER_TOKENS]
            if len(toks) < 2:
                continue
            h_real = _hit_fraction(toks, ref_bigrams)
            h_shuf = _shuffled_hit_fraction(
                toks, ref_bigrams, random.Random(seed + n_docs + j)
            )
            target_h_real.append(h_real)
            target_ratios.append(_order_ratio(h_real, h_shuf))

    order_test_enabled = bool(target_ratios)
    ratio_threshold = ORDER_RATIO_THRESHOLD
    h_real_floor = 0.0
    if order_test_enabled:
        below = sum(1 for r in target_ratios if r < ORDER_RATIO_THRESHOLD)
        if below > ORDER_GUARD_FRAC * len(target_ratios):
            ratio_threshold = min(
                ORDER_RATIO_THRESHOLD, 0.5 * _percentile(target_ratios, 0.05)
            )
        h_real_floor = ORDER_HREAL_REL * _median(target_h_real)

    # ---- pass 1: pool unigram counts (whole pool) + exact dedupe -------------
    pool_counts = Counter()
    seen = set()
    unique = []
    for i, text in enumerate(docs):
        toks = text.lower().split()
        pool_counts.update(toks)
        key = hashlib.blake2b(
            " ".join(toks).encode("utf-8", "replace"), digest_size=16
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(i)
    del seen

    # ---- unigram log-likelihood ratio table ----------------------------------
    n_target = sum(target_counts.values())
    n_pool = sum(pool_counts.values())
    vocab = len(pool_counts) + sum(1 for w in target_counts if w not in pool_counts)
    llr = {}
    unseen_llr = 0.0
    if n_target > 0:
        mu = max(1.0, TARGET_MU_FRAC * n_target)
        pool_den = n_pool + POOL_ALPHA * max(vocab, 1)
        tgt_den = n_target + mu
        unseen_llr = math.log(mu / tgt_den)
        for w, c in target_counts.items():
            p_pool = (pool_counts.get(w, 0) + POOL_ALPHA) / pool_den
            llr[w] = math.log((c + mu * p_pool) / (tgt_den * p_pool))
    del pool_counts

    # ---- pass 2: junk filters + scoring --------------------------------------
    get = llr.get
    scored = []
    for i in unique:
        text = docs[i]
        if len(text) < MIN_CHARS:
            continue
        if _alpha_space_fraction(text) < MIN_ALPHA_SPACE:
            continue
        if _zlib_ratio(text) < MIN_ZLIB_RATIO:
            continue

        all_toks = text[:SCORE_CHARS].lower().split()
        if not all_toks:
            continue

        if order_test_enabled:
            otoks = all_toks[:ORDER_TOKENS]
            h_real = _hit_fraction(otoks, ref_bigrams)
            if h_real < h_real_floor:
                continue
            h_shuf = _shuffled_hit_fraction(otoks, ref_bigrams, random.Random(seed + i))
            if _order_ratio(h_real, h_shuf) < ratio_threshold:
                continue

        toks = all_toks[:SCORE_TOKENS]
        score = sum(map(get, toks, repeat(unseen_llr))) / len(toks)
        if _HEADER_RE.match(text) or " @-@ " in text:
            score += HEADER_BONUS
        scored.append((-score, i))

    if not scored:
        idx = list(unique)
        random.Random(seed).shuffle(idx)
        return idx

    scored.sort()
    
    core = []
    filler = []
    for _, i in scored:
        text = docs[i]
        if _HEADER_RE.match(text) or " @-@ " in text:
            core.append(i)
        else:
            filler.append(i)

    # ---- selection: block-stratified order (10 buckets) --------------------
    scores = [item[0] for item in scored]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        buckets = [scored]
    else:
        # Create 10 buckets based on score range
        buckets = [[] for _ in range(10)]
        for item in scored:
            # item[0] is negative score (smaller is better/higher score)
            # We want highest score first (smallest negative score)
            # Normalize item[0] from [min_s, max_s] to [0, 9]
            idx = int(((item[0] - min_s) / (max_s - min_s)) * 9)
            buckets[idx].append(item)
        # The way we want them is from highest score (closest to 0) to lowest.
        # The logic above puts highest score (min_s) at bucket 0.
        # So we need to reverse the buckets to get "best" first.
        buckets = buckets[::-1]
        # Re-sort buckets slightly to ensure strict order if needed
        # (They are already sorted by the score range)

    # Block-stratified shuffle
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    
    # Flatten into a list of scores/indices
    flat_scored = []
    for b in buckets:
        flat_scored.extend(b)

    # Extract IDs maintaining the stratified block order
    stratified_ids = [item[1] for item in flat_scored]
    
    # Re-separate Core and Filler while preserving order
    core_ordered = [i for i in stratified_ids if _HEADER_RE.match(docs[i]) or " @-@ " in docs[i]]
    filler_ordered = [i for i in stratified_ids if not (_HEADER_RE.match(docs[i]) or " @-@ " in docs[i])]

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pass

    def get_counts(indices):
        nonlocal seen_chars, seen_tokens
        chunk_counts = []
        if enc is not None and (time.time() - t0) < TIME_BUDGET:
            try:
                chunk_counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in indices])]
                seen_chars += sum(len(docs[i]) for i in indices)
                seen_tokens += sum(chunk_counts)
            except Exception:
                pass
        if not chunk_counts:
            ratio = (seen_tokens / seen_chars) if seen_chars > 0 else 0.25
            chunk_counts = [max(1, int(len(docs[i]) * ratio)) for i in indices]
        return chunk_counts

    selected = []
    total = 0
    seen_chars = 0
    seen_tokens = 0
    
    # 1. Core once
    if core_ordered:
        counts = get_counts(core_ordered)
        for i, c in zip(core_ordered, counts):
            selected.append(i)
            total += c
            if total >= T: break
            
    # 2. Core twice
    if total < T and core_ordered:
        counts = get_counts(core_ordered)
        for i, c in zip(core_ordered, counts):
            selected.append(i)
            total += c
            if total >= T: break

    # 3. Filler
    if total < T and filler_ordered:
        for i in filler_ordered:
            counts = get_counts([i])
            c = counts[0]
            selected.append(i)
            total += c
            if total >= T: break

    return selected
