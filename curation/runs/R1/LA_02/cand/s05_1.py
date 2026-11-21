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
    rng = random.Random(seed)
    
    target_counts = Counter()
    bigram_counts = Counter()
    target_alpha_words = Counter()
    
    for text in target_sample:
        words_raw = text.lower().split()
        target_counts.update(words_raw)
        bigram_counts.update(zip(words_raw, words_raw[1:]))
        alpha_words = [re.sub(r'[^\w]', '', w).lower() for w in words_raw]
        target_alpha_words.update(alpha_words)

    ref_bigrams = {bg for bg, c in bigram_counts.items() if c >= ORDER_REF_MIN_COUNT}
    
    pool_counts = Counter()
    pool_alpha_counts = Counter()
    seen_exact = set()
    unique_indices = []
    
    for i, text in enumerate(docs):
        toks = text.lower().split()
        pool_counts.update(toks)
        alpha_toks = [re.sub(r'[^\w]', '', w).lower() for w in toks]
        pool_alpha_counts.update(alpha_toks)
        key = hashlib.blake2b(" ".join(toks).encode("utf-8", "replace"), digest_size=16).digest()
        if key not in seen_exact:
            seen_exact.add(key)
            unique_indices.append(i)

    V = set()
    for w, c in pool_alpha_counts.items():
        if c >= 5: V.add(w)
    for w in target_alpha_words:
        V.add(w)

    # Calibration
    oov_list = []
    rep_list = []
    for text in target_sample:
        words_raw = text.lower().split()
        toks_noise = words_raw[:600]
        oov_c = 0
        rep_c = 0
        for i, w in enumerate(toks_noise):
            if len(w) >= 3 and w[0].isalpha() and w[0].islower() and w not in V:
                oov_c += 1
            if i > 0 and toks_noise[i] == toks_noise[i-1] and toks_noise[i].isalpha():
                rep_c += 1
        n = len(toks_noise)
        oov_list.append(oov_c / n if n > 0 else 0)
        rep_list.append(rep_c / n if n > 0 else 0)

    oov_thresh = max(2 * _percentile(oov_list, 0.95), _percentile(oov_list, 0.95) + 0.02)
    rep_thresh = max(3 * _percentile(rep_list, 0.95), 0.01)

    # LLR
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

    scored = []
    for i in unique_indices:
        text = docs[i]
        if len(text) < MIN_CHARS or _alpha_space_fraction(text) < MIN_ALPHA_SPACE or _zlib_ratio(text) < MIN_ZLIB_RATIO:
            continue
        all_toks = text[:SCORE_CHARS].lower().split()
        if not all_toks: continue
        noise_toks = all_toks[:600]
        oov_c = 0
        rep_c = 0
        for j, w in enumerate(noise_toks):
            if len(w) >= 3 and w[0].isalpha() and w[0].islower() and w not in V:
                oov_c += 1
            if j > 0 and noise_toks[j] == noise_toks[j-1] and noise_toks[j].isalpha():
                rep_c += 1
        n_n = len(noise_toks)
        if n_n > 0 and ((oov_c / n_n) > oov_thresh or (rep_c / n_n) > rep_thresh):
            continue
        
        score = sum(map(lambda w: llr.get(w, unseen_llr), all_toks[:SCORE_TOKENS])) / len(all_toks[:SCORE_TOKENS])
        if _HEADER_RE.match(text) or " @-@ " in text:
            score += HEADER_BONUS
        
        alpha_only = [re.sub(r'[^\w]', '', w).lower() for w in all_toks[:300] if w.strip()]
        shingles = [" ".join(alpha_only[k:k+5]) for k in range(len(alpha_only)-4)]
        sketch = sorted(list(set(hashlib.md5(s.encode()).digest() for s in shingles)))[:8]
        scored.append((-score, i, sketch))

    scored.sort()
    final_ranked = []
    inverted_index = {}
    for _, i, sketch in scored:
        match_count = 0
        for s in sketch:
            if s in inverted_index:
                match_count += 1
        if match_count < 5:
            final_ranked.append(i)
            for s in sketch:
                if s not in inverted_index: inverted_index[s] = []
                inverted_index[s].append(i)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        enc = None

    selected = []
    total_tokens = 0
    for i in final_ranked:
        doc = docs[i]
        if enc and (time.time() - t0) < TIME_BUDGET:
            count = len(enc.encode_ordinary(doc))
        else:
            count = int(len(doc) * 0.25)
        if total_tokens + count <= T:
            selected.append(i)
            total_tokens += count
            
    return selected
