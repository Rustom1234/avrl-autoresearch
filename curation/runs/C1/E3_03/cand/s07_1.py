"""Curator: uses target-fidelity scores via weighted random sampling to select docs.

Pipeline
  1. Calculate features and calibrate hit thresholds on target_sample.
  2. Filter pool docs using shape, spam, and bigram-density constraints.
  3. Score survivors based on bigram density with the target sample.
  4. Use Weighted Random Sampling (with power-scaling) to fill T tokens.
"""
import os
import random
import re
import hashlib
import math
from collections import Counter
import numpy as np

try:
    from common import T
except Exception:
    T = 9_830_400

SPAM_LITERALS = (
    "discount code",
    "coupon",
    "free shipping",
    "buy now",
    "for sale |",
    "best price guaranteed",
)
SPAM_ORDER_RE = re.compile(r"order\b[^\n]{0,40}?\bonline")
COMMON_BIGRAMS = frozenset([
    ("of", "the"), ("in", "the"), ("to", "the"),
    ("and", "the"), ("on", "the"), (",", "and"), (".", "the"),
])

LONG_TOKEN = 25
MIN_WORDS_FOR_COMMON = 100

def _basic_features(text, words):
    n = len(words)
    space_ratio = text.count(" ") / max(1, len(text))
    lens = [len(w) for w in words]
    mean_len = sum(lens) / n
    long_frac = sum(1 for l in lens if l > LONG_TOKEN) / n
    return space_ratio, mean_len, long_frac

def _spam_flag(low, n_words):
    hits = 0
    for phrase in SPAM_LITERALS:
        hits += low.count(phrase)
    hits += len(SPAM_ORDER_RE.findall(low))
    return hits >= 2 or (hits > 0 and hits * 200 > n_words)

def _count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        out = []
        chunk = 2000
        for s in range(0, len(texts), chunk):
            batch = enc.encode_ordinary_batch(texts[s : s + chunk], num_threads=8)
            out.extend(len(b) for b in batch)
        return out
    except Exception:
        return [max(1, len(t) // 4) for t in texts]

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    t_bigrams = []
    t_feats = []
    for t in target_sample:
        low = t.lower()
        words = low.split()
        if len(words) < 2: continue
        t_bigrams.append(list(zip(words, words[1:])))
        t_feats.append(_basic_features(t, words) + (len(words),))

    doc_freq = Counter()
    for bg in t_bigrams:
        doc_freq.update(set(bg))
    target_set = set(doc_freq)

    calibrated = len(t_bigrams) >= 5
    if calibrated:
        loo_scores = []
        common_rates = []
        for bg, feats in zip(t_bigrams, t_feats):
            hits = sum(1 for b in bg if doc_freq[b] > 1)
            loo_scores.append(hits / len(bg))
            n_words = feats[3]
            if n_words >= MIN_WORDS_FOR_COMMON:
                common_rates.append(100.0 * sum(map(COMMON_BIGRAMS.__contains__, bg)) / n_words)
        hit_thr = float(np.percentile(loo_scores, 2))
        common_thr = float(np.percentile(common_rates, 2)) if common_rates else 0.0
        space_vals = [f[0] for f in t_feats]
        mean_vals = [f[1] for f in t_feats]
        long_vals = [f[2] for f in t_feats]
        space_lo, space_hi = min(space_vals), max(space_vals)
        mean_lo, mean_hi = min(mean_vals) * 0.9, max(mean_vals) * 1.1
        long_hi = max(long_vals) + 0.002
    else:
        hit_thr, common_thr = 0.0, 0.0
        space_lo, space_hi = 0.05, 0.35
        mean_lo, mean_hi = 2.0, 9.0
        long_hi = 0.02

    scores = []
    indices = []
    seen_hashes = set()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text: continue
        low = text.lower()
        words = low.split()
        n = len(words)
        if n < 2: continue

        sp_r, m_l, l_f = _basic_features(text, words)
        if sp_r < space_lo or sp_r > space_hi or l_f > long_hi or m_l < mean_lo or m_l > mean_hi:
            continue
        if _spam_flag(low, n): continue

        bg = list(zip(words, words[1:]))
        hit_frac = sum(map(target_set.__contains__, bg)) / len(bg)
        if hit_frac < hit_thr: continue
        if n >= MIN_WORDS_FOR_COMMON:
            if 100.0 * sum(map(COMMON_BIGRAMS.__contains__, bg)) / n < common_thr:
                continue

        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen_hashes: continue
        seen_hashes.add(h)

        scores.append(hit_frac)
        indices.append(i)

    if not indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    scores = np.array(scores)
    scores = np.power(scores, 2.0)
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    
    token_counts = _count_tokens([docs[i] for i in indices])
    
    final_indices = []
    total_tokens = 0
    
    while total_tokens < T:
        # Replace rng.choices(p=...) with manual weighted sampling
        r = rng.random()
        cumulative_probs = np.cumsum(probs)
        idx_pos = np.searchsorted(cumulative_probs, r)
        idx_pos = min(idx_pos, len(indices) - 1)
        
        idx = indices[idx_pos]
        c = token_counts[idx_pos]
        final_indices.append(idx)
        total_tokens += c
        
        if len(final_indices) > 100000: break 

    return final_indices
