"""Curator: dedupe, drop obvious junk, rank by 2-gram LLR, fill to T tokens.
    Core docs (WikiText style) appear twice before filler docs.
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
    if n < 1:
        return 0.0
    return sum(map(ref.__contains__, zip(toks, toks[1:]))) / n

def _shuffled_hit_fraction(toks, ref, rng):
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

    # ---- target statistics (unigram & bigram) ---------------------------------
    target_unigrams = Counter()
    target_bigrams = Counter()
    for text in target_sample:
        toks = text.lower().split()
        target_unigrams.update(toks)
        target_bigrams.update(zip(toks, toks[1:]))
    
    ref_bigrams = {bg for bg, c in target_bigrams.items() if c >= ORDER_REF_MIN_COUNT}
    
    # ---- calibrate the order test -----------------------------------------------
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
            ratio_threshold = min(ORDER_RATIO_THRESHOLD, 0.5 * _percentile(target_ratios, 0.05))
        h_real_floor = ORDER_HREAL_REL * _median(target_h_real)

    # ---- pool statistics ---------------------------------------------------------
    pool_unigrams = Counter()
    pool_bigrams = Counter()
    seen = set()
    unique = []
    for i, text in enumerate(docs):
        toks = text.lower().split()
        pool_unigrams.update(toks)
        pool_bigrams.update(zip(toks, toks[1:]))
        key = hashlib.blake2b(" ".join(toks).encode("utf-8", "replace"), digest_size=16).digest()
        if key not in seen:
            seen.add(key)
            unique.append(i)
    
    # ---- LLR calculation ---------------------------------------------------------
    n_target_unigrams = sum(target_unigrams.values())
    n_target_bigrams = sum(target_bigrams.values())
    vocab_size = len(pool_unigrams) + sum(1 for w in target_unigrams if w not in pool_unigrams)
    
    # Unigram LLR parameters
    mu_unig = max(1.0, TARGET_MU_FRAC * n_target_unigrams)
    pool_den_unig = sum(pool_unigrams.values()) + POOL_ALPHA * max(vocab_size, 1)
    tgt_den_unig = n_target_unigrams + mu_unig
    unseen_llr_unig = math.log(mu_unig / tgt_den_unig)
    
    # Bigram LLR parameters
    mu_big = max(1.0, TARGET_MU_FRAC * n_target_bigrams)
    pool_den_big = sum(pool_bigrams.values()) + POOL_ALPHA * (vocab_size * vocab_size)
    tgt_den_big = n_target_bigrams + mu_big
    unseen_llr_big = math.log(mu_big / tgt_den_big)

    llr_map = {}
    for w in target_unigrams:
        p_pool = (pool_unigrams.get(w, 0) + POOL_ALPHA) / pool_den_unig
        llr_map[w] = math.log((target_unigrams[w] + mu_unig * p_pool) / (tgt_den_unig * p_pool))
    
    # Pre-calculate bigram LLRs
    bigram_llr_map = {}
    for bg in target_bigrams:
        p_pool = (pool_bigrams.get(bg, 0) + POOL_ALPHA) / pool_den_big
        bigram_llr_map[bg] = math.log((target_bigrams[bg] + mu_big * p_pool) / (tgt_den_big * p_pool))

    # ---- Scoring ------------------------------------------------------------------
    get_unig = llr_map.get
    get_big = bigram_llr_map.get
    scored = []
    for i in unique:
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
        score = 0.0
        if len(toks) >= 2:
            # Use bigram LLR for the first (len-1) pairs, fall back to unigram LLR for the last token
            bigram_score = 0.0
            for k in range(len(toks) - 1):
                bg = (toks[k], toks[k+1])
                bigram_score += get_big(bg, unseen_llr_big)
            score = bigram_score / (len(toks) - 1)
        else:
            score = sum(map(get_unig, toks, repeat(unseen_llr_unig))) / len(toks)
        
        if _HEADER_RE.match(text) or " @-@ " in text:
            score += HEADER_BONUS
        scored.append((-score, i))

    if not scored:
        idx = list(unique)
        random.Random(seed).shuffle(idx)
        return idx

    scored.sort()
    core = [i for _, i in scored if _HEADER_RE.match(docs[i]) or " @-@ " in docs[i]]
    filler = [i for _, i in scored if not (_HEADER_RE.match(docs[i]) or " @-@ " in docs[i])]

    # ---- Selection -----------------------------------------------------------------
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except: pass

    def get_counts(indices):
        nonlocal seen_chars, seen_tokens
        if time.time() - t0 > TIME_BUDGET:
            # Fallback estimation
            ratio = (seen_tokens / seen_chars) if seen_chars > 0 else 0.25
            return [max(1, int(len(docs[i]) * ratio)) for i in indices]
        try:
            batch = [docs[i] for i in indices]
            counts = [len(x) for x in enc.encode_ordinary_batch(batch)]
            seen_chars += sum(len(docs[i]) for i in indices)
            seen_tokens += sum(counts)
            return counts
        except:
            return [max(1, int(len(docs[i]) * 0.25)) for i in indices]

    selected = []
    total_t = 0
    seen_chars = 0
    seen_tokens = 0
    
    for _ in range(2): # Core twice
        if core:
            counts = get_counts(core)
            for i, c in zip(core, counts):
                selected.append(i)
                total_t += c
                if total_t >= T: break
            if total_t >= T: break
            
    if total_t < T and filler:
        for i in filler:
            counts = get_counts([i])
            c = counts[0]
            selected.append(i)
            total_t += c
            if total_t >= T: break

    random.Random(seed).shuffle(selected)
    return selected
