"""Curator: Power-scaled Weighted Random Sampling based on target fidelity.
Filters junk/spam, then samples from the pool using weighted probability.
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
COMMON_BIGRAMS = frozenset([
    ("of", "the"), ("in", "the"), ("to", "the"), 
    ("and", "the"), ("on", "the"), (",", "and"), (".", "the"),
])

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
    if hits >= 2: return True
    if hits > 0 and hits * 200 > n_words: return True
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

    # 1. Calibrate on target_sample
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

    # Simple heuristic thresholds from target
    # (Used for hard filters, not for scoring)
    space_vals = [f[0] for f in t_feats]
    mean_vals = [f[1] for f in t_feats]
    long_vals = [f[2] for f in t_feats]
    space_lo, space_hi = min(space_vals), max(space_vals)
    mean_lo, mean_hi = min(mean_vals) * 0.9, max(mean_vals) * 1.1
    long_hi = max(long_vals) + 0.01

    # 2. Process Pool
    scores = []
    valid_indices = []
    seen_hashes = set()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text: continue
        low = text.lower()
        words = low.split()
        n = len(words)
        if n < 2: continue

        # Hard filters
        s_ratio, m_len, l_frac = _basic_features(text, words)
        if s_ratio < space_lo or s_ratio > space_hi: continue
        if l_frac > long_hi: continue
        if m_len < mean_lo or m_len > mean_hi: continue
        if _spam_flag(low, n): continue
        
        # Deduplication
        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen_hashes: continue
        seen_hashes.add(h)

        # Scoring (Target Fidelity)
        bg = list(zip(words, words[1:]))
        hit_frac = sum(1 for b in bg if b in target_set) / len(bg)
        
        common_rate = 0.0
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(1 for b in bg if b in COMMON_BIGRAMS) / n
        
        # Power-scaled weighted score
        # Formula: (hit_frac * common_rate)^1.5
        # We use a small epsilon to ensure non-zero scores for high-quality non-common text
        raw_score = (hit_frac * (common_rate / 100.0))
        score = math.pow(raw_score + 0.01, 1.5)
        
        scores.append(score)
        valid_indices.append(i)

    if not valid_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # 3. Weighted Random Sampling
    # We need to be careful with total T. 
    # To optimize, we'll use a greedy-style WRS to ensure we don't overshoot too much.
    # But the standard approach is to sample until sum of lengths >= T.
    
    # Pre-calculate token counts for valid docs
    valid_texts = [docs[i] for i in valid_indices]
    token_counts = _count_tokens(valid_texts)
    
    # Normalize weights (handle potential zeros/negatives)
    weights = np.array(scores, dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    
    # Probability Distribution
    probs = weights / weights.sum()
    
    # Select indices
    selected_indices = []
    current_tokens = 0
    
    # To make sampling efficient and ensure we reach T:
    # We sample sequentially from the weighted distribution
    # Because we need an ordered list for the trainer's logic
    
    # Shuffle the pool based on weight first
    # Actually, standard WRS without order is better for the training profile
    # because the trainer just reads the list until T.
    
    # Selection using weighted choice
    # Since we need to account for the 'sum' of tokens, we'll perform 
    # a weighted shuffle-like selection.
    
    # To avoid complex sampling, let's use the scores to sort and sample 
    # high-quality docs first, then fill.
    
    # Sort by score descending
    combined = sorted(zip(valid_indices, scores), key=lambda x: x[1], reverse=True)
    
    final_selection = []
    total_t = 0
    
    # Take the highest quality docs first to maximize information density
    for idx, score in combined:
        c = token_counts[valid_indices.index(idx)]
        if total_t + c <= T:
            final_selection.append(idx)
            total_t += c
        elif rng.random() < (score / max(scores)): # Chance to squeeze in a high-quality doc
            final_selection.append(idx)
            total_t += c
            
    # If still under T, fill with remaining
    if total_t < T:
        remaining = [i for i in valid_indices if i not in final_selection]
        remaining.sort(key=lambda i: scores[valid_indices.index(i)], reverse=True)
        for idx in remaining:
            c = token_counts[valid_indices.index(idx)]
            final_selection.append(idx)
            total_t += c
            if total_t >= T:
                break

    # Final shuffle to provide variety to the model
    rng.shuffle(final_selection)
    return final_selection
