"""Curator: dedupe, drop junk, rank by LLR, and use coherence-based repetition.
    Pairs core docs with similar neighbors for variety in repeated content.
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
ORDER_TOKENS = 600
ORDER_REF_MIN_COUNT = 3
ORDER_SHUFFLES = 2
ORDER_EPS = 0.005
ORDER_RATIO_THRESHOLD = 2.0
ORDER_GUARD_FRAC = 0.05
ORDER_HREAL_REL = 0.3
SCORE_TOKENS = 1000
SCORE_CHARS = 16000
HEADER_BONUS = 1.0
POOL_ALPHA = 0.5
TARGET_MU_FRAC = 0.05
TOKENISE_CHUNK = 1000
TIME_BUDGET = 150.0

_HEADER_RE = re.compile(r"^\s*= [^=\n]+ =\s*\n")

def _hit_fraction(toks, ref):
    n = len(toks) - 1
    if n < 1: return 0.0
    return sum(map(ref.__contains__, zip(toks, toks[1:]))) / n

def _shuffled_hit_fraction(toks, ref, rng):
    n = len(toks) - 1
    if n < 1: return 0.0
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
    if not values: return 0.0
    v = sorted(values)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])

def _percentile(values, q):
    if not values: return 0.0
    v = sorted(values)
    if len(v) == 1: return v[0]
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    frac = pos - lo
    return v[lo] * (1.0 - frac) + v[hi] * frac

def _alpha_space_fraction(text):
    n = len(text)
    if n == 0: return 0.0
    return (sum(map(str.isalpha, text)) + sum(map(str.isspace, text))) / n

def _zlib_ratio(text):
    raw = text.encode("utf-8", "replace")
    if not raw: return 0.0
    return len(zlib.compress(raw)) / len(raw)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0: return []

    # ---- target statistics --------------------------------------------------
    target_counts = Counter()
    bigram_counts = Counter()
    for text in target_sample:
        toks = text.lower().split()
        target_counts.update(toks)
        bigram_counts.update(zip(toks, toks[1:]))
    ref_bigrams = {bg for bg, c in bigram_counts.items() if c >= ORDER_REF_MIN_COUNT}

    # ---- calibrate order test ------------------------------------------------
    target_h_real, target_ratios = [], []
    if ref_bigrams:
        for j, text in enumerate(target_sample):
            toks = text[:SCORE_CHARS].lower().split()[:ORDER_TOKENS]
            if len(toks) < 2: continue
            h_real = _hit_fraction(toks, ref_bigrams)
            h_shuf = _shuffled_hit_fraction(toks, ref_bigrams, random.Random(seed + n_docs + j))
            target_h_real.append(h_real)
            target_ratios.append(_order_ratio(h_real, h_shuf))

    order_test_enabled = bool(target_ratios)
    ratio_threshold = ORDER_RATIO_THRESHOLD
    h_real_floor = 0.0
    if order_test_enabled:
        below = sum(1 for r in target_ratios if r < ORDER_RATIO_THRESHOLD)
        if below > ORDER_GUARD_FRAC * len(target_ratios):
            ratio_threshold = min(ORDER_RATIO_THRESHOLD, 0.5 * _percentile(target_ratios, 0.05))
        h_real_floor = ORDER_HREAL_REL * _median(target_h_real)

    # ---- pass 1: pool unigram counts ----------------------------------------
    pool_counts = Counter()
    seen = set()
    unique_indices = []
    for i, text in enumerate(docs):
        toks = text.lower().split()
        pool_counts.update(toks)
        key = hashlib.blake2b(" ".join(toks).encode("utf-8", "replace"), digest_size=16).digest()
        if key not in seen:
            seen.add(key)
            unique_indices.append(i)

    # ---- LLR table ------------------------------------------------------------
    n_target = sum(target_counts.values())
    n_pool = sum(pool_counts.values())
    vocab_size = len(pool_counts) + sum(1 for w in target_counts if w not in pool_counts)
    llr = {}
    unseen_llr = 0.0
    if n_target > 0:
        mu = max(1.0, TARGET_MU_FRAC * n_target)
        pool_den = n_pool + POOL_ALPHA * max(vocab_size, 1)
        tgt_den = n_target + mu
        unseen_llr = math.log(mu / tgt_den)
        for w, c in target_counts.items():
            p_pool = (pool_counts.get(w, 0) + POOL_ALPHA) / pool_den
            llr[w] = math.log((c + mu * p_pool) / (tgt_den * p_pool))

    # ---- pass 2: scoring -----------------------------------------------------
    scored = []
    get = llr.get
    for i in unique_indices:
        text = docs[i]
        if len(text) < MIN_CHARS or _alpha_space_fraction(text) < MIN_ALPHA_SPACE or _zlib_ratio(text) < MIN_ZLIB_RATIO:
            continue
        
        all_toks = text[:SCORE_CHARS].lower().split()
        if not all_toks: continue

        if order_test_enabled:
            otoks = all_toks[:ORDER_TOKENS]
            h_real = _hit_fraction(otoks, ref_bigrams)
            if h_real < h_real_floor: continue
            h_shuf = _shuffled_hit_fraction(otoks, ref_bigrams, random.Random(seed + i))
            if _order_ratio(h_real, h_shuf) < ratio_threshold: continue

        toks = all_toks[:SCORE_TOKENS]
        score = sum(map(get, toks, repeat(unseen_llr))) / len(toks)
        if _HEADER_RE.match(text) or " @-@ " in text:
            score += HEADER_BONUS
        scored.append((-score, i))

    if not scored:
        res = list(unique_indices)
        random.Random(seed).shuffle(res)
        return res

    scored.sort()
    core, filler = [], []
    for _, i in scored:
        if _HEADER_RE.match(docs[i]) or " @-@ " in docs[i]:
            core.append(i)
        else:
            filler.append(i)

    # ---- Coherence Preparation -------------------------------------------------
    # Pre-calculate LLR vectors for core docs for similarity
    core_vectors = []
    for i in core:
        toks = docs[i][:SCORE_CHARS].lower().split()[:SCORE_TOKENS]
        # Only include words with non-zero LLR (meaning they are in target or shared)
        vec = [llr.get(w, unseen_llr) for w in toks]
        # Standardize vector length for basic cosine-like similarity
        core_vectors.append((i, vec))

    # ---- Selection logic -------------------------------------------------------
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        enc = None

    def get_counts(indices):
        if not indices: return []
        chunk_counts = []
        if enc and (time.time() - t0) < TIME_BUDGET:
            try:
                batch_toks = [docs[i] for i in indices]
                # Process in chunks to avoid memory issues or overflow
                chunk_count = 0
                for batch in range(0, len(batch_toks), 100):
                    chunk_count += len(enc.encode_ordinary_batch(batch_toks[batch:batch+100]))
                chunk_counts = [chunk_count // len(indices)] # Approximation
                # Re-calculating precisely is hard with varying lengths, so we sum and average
                total_t = 0
                for b in range(0, len(batch_toks), 100):
                    total_t += sum(len(x) for x in enc.encode_ordinary_batch(batch_toks[b:b+100]))
                chunk_counts = [total_t // len(indices)]
            except: pass
        if not chunk_counts:
            # Fallback to char ratio
            avg_ratio = 0.25
            # Simple estimation for the whole batch
            return [max(1, int(len(docs[i]) * avg_ratio)) for i in indices]
        return chunk_counts

    selected = []
    total_tokens = 0
    
    # 1. Core once
    if core:
        # Calculate weights for similarity
        # We want to find the best "neighbor" for each core doc
        # To keep it O(N^2) manageable, we limit the pool to the top 100 core docs if needed
        # or just use a small subset of fillers.
        core_indices = core
        counts = get_counts(core_indices)
        for i, c in zip(core_indices, counts):
            selected.append(i)
            total_tokens += c
            if total_tokens >= T: break

    if total_tokens < T and core:
        # 2. Core repeated (Coherence-based)
        # For each core doc, find a similar neighbor in the top of the 'filler' list
        # using the LLR vectors.
        filler_vectors = []
        # Only look at top 200 filler docs to save time
        for i in filler[:200]:
            toks = docs[i][:SCORE_CHARS].lower().split()[:SCORE_TOKENS]
            filler_vectors.append((i, [llr.get(w, unseen_llr) for w in toks]))
            
        for i_idx, vec_i in core_vectors:
            if total_tokens >= T: break
            
            best_neighbor = i_idx # Default to self
            max_sim = -float('inf')
            
            # Find most similar filler
            for f_idx, vec_f in filler_vectors:
                # Simple dot product as a proxy for similarity
                sim = sum(a * b for a, b in zip(vec_i, vec_f))
                if sim > max_sim:
                    max_sim = sim
                    best_neighbor = f_idx
            
            selected.append(best_neighbor)
            # Approximate count (since we don't have exact per-item count easily in batch)
            total_tokens += int(len(docs[best_neighbor]) * 0.25)
            if total_tokens >= T: break

    if total_tokens < T and filler:
        # 3. Remaining Filler
        for i in filler:
            c = int(len(docs[i]) * 0.25)
            selected.append(i)
            total_tokens += c
            if total_tokens >= T: break

    random.Random(seed).shuffle(selected)
    return selected
