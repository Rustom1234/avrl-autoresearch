"""Curator: Hard junk filters with Target-Similarity scoring and Weighted Random Sampling."""
import os
import random
import re
import hashlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
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

COMMON_BIGRAMS = frozenset(
    [
        ("of", "the"),
        ("in", "the"),
        ("to", "the"),
        ("and", "the"),
        ("on", "the"),
        (",", "and"),
        (".", "the"),
    ]
)

LONG_TOKEN = 25
MIN_WORDS_FOR_COMMON = 100

def _basic_features(text, words):
    n = len(words)
    space_ratio = text.count(" ") / max(1, len(text))
    lens = list(map(len, words))
    mean_len = sum(lens) / n
    long_frac = sum(1 for length in lens if length > LONG_TOKEN) / n
    return space_ratio, mean_len, long_frac

def _spam_flag(low, n_words):
    hits = 0
    for phrase in SPAM_LITERALS:
        hits += low.count(phrase)
    hits += len(SPAM_ORDER_RE.findall(low))
    if hits >= 2:
        return True
    if hits > 0 and hits * 200 > n_words:
        return True
    return False

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

    # Calibration
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
                common = sum(map(COMMON_BIGRAMS.__contains__, bg))
                common_rates.append(100.0 * common / n_words)
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

    # Filtering and Scoring
    seen = set()
    candidates = [] # (index, score, tokens)
    
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text: continue
        low = text.lower()
        words = low.split()
        n = len(words)
        if n < 2: continue

        s_ratio, m_len, l_frac = _basic_features(text, words)
        if s_ratio < space_lo or s_ratio > space_hi or l_frac > long_hi or m_len < mean_lo or m_len > mean_hi:
            continue
        if _spam_flag(low, n): continue

        bg = list(zip(words, words[1:]))
        hit_frac = sum(map(target_set.__contains__, bg)) / len(bg)
        if hit_frac < hit_thr: continue
        
        common_rate = 0.0
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(map(COMMON_BIGRAMS.__contains__, bg)) / n
            if common_rate < common_thr: continue
        
        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen: continue
        seen.add(h)

        score = (hit_frac * (common_rate / 100.0)) ** 1.5
        score = max(score, 1e-6)
        
        token_count = _count_tokens([text])[0]
        candidates.append((i, score, token_count))

    if not candidates:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    weights = np.array([c[1] for c in candidates])
    sum_w = weights.sum()
    probs = weights / sum_w
    
    pool = candidates
    selected_indices = []
    current_tokens = 0
    
    # Manual weighted sampling since random.choices(p=...) isn't available in some Python versions/objects
    # However, the error was specifically on the object passed (random.Random vs random.random)
    # We'll use the standard way to sample from a distribution:
    while current_tokens < T:
        # Use a binary search approach or a simple cumulative sum for efficiency
        r = rng.random() * sum_w
        cumulative = 0
        idx_choice = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                idx_choice = i
                break
        
        doc_idx, score, count = pool[idx_choice]
        selected_indices.append(doc_idx)
        current_tokens += count
        
        # Break if we are looping too much (infinite loop protection)
        if len(selected_indices) > 200000:
            break

    if current_tokens < T:
        best_docs = sorted(pool, key=lambda x: x[1], reverse=True)
        for d_idx, score, count in best_docs:
            selected_indices.append(d_idx)
            current_tokens += count
            if current_tokens >= T:
                break
                
    rng.shuffle(selected_indices)
    return selected_indices
