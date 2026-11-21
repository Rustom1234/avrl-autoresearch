"""Curator: refined filtering with strict wiki checks and scrambled-text detection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
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

    t_bigrams = []
    t_feats = []
    for t in target_sample:
        low = t.lower()
        words = low.split()
        if len(words) < 2:
            continue
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

    in_target = target_set.__contains__
    in_common = COMMON_BIGRAMS.__contains__
    seen = set()
    group_a, group_b = [], []

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        low = text.lower()
        words = low.split()
        n = len(words)
        if n < 2:
            continue

        space_ratio, mean_len, long_frac = _basic_features(text, words)
        if space_ratio < space_lo or space_ratio > space_hi:
            continue
        if long_frac > long_hi:
            continue
        if mean_len < mean_lo or mean_len > mean_hi:
            continue

        if _spam_flag(low, n):
            continue

        bg = list(zip(words, words[1:]))
        hit_frac = sum(map(in_target, bg)) / len(bg)
        
        # Scrambled-text filter: High common rate + extremely low target hit rate
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(map(in_common, bg)) / n
            if hit_frac < 0.05 and common_rate > 10.0:
                continue
        
        if hit_frac < hit_thr:
            continue
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(map(in_common, bg)) / n
            if common_rate < common_thr:
                continue

        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)

        # Strict Wiki Check
        is_wiki = (
            text.strip().startswith("= ")
            and "@-@" in text
            and text.count("@-@") > 1
        )
        (group_a if is_wiki else group_b).append(i)

    if not group_a and not group_b:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    rng = random.Random(seed)
    rng.shuffle(group_a)
    rng.shuffle(group_b)

    # Logic: Repetition of A first, then B
    res = []
    total_tokens = 0
    
    # Helper to add docs and track tokens
    def add_docs(indices_list):
        nonlocal total_tokens
        for idx in indices_list:
            res.append(idx)
            total_tokens += max(1, len(docs[idx].split())) // 4 # Fast approx for loop
            if total_tokens >= T: return True
        return False

    # Add group A multiple times if needed
    while total_tokens < T and group_a:
        if add_docs(group_a): break
        
    # Fill remaining with B
    if total_tokens < T:
        chunk = 2000
        for i in range(0, len(group_b), chunk):
            batch = group_b[i : i + chunk]
            # Use real count for the final check
            batch_counts = _count_tokens([docs[j] for j in batch])
            for idx, count in zip(batch, batch_counts):
                res.append(idx)
                total_tokens += count
                if total_tokens >= T:
                    return res[:int(T // (max(1, len(docs[res[0]]) // 4)))] # Adjust cut

    # Final cleanup / re-shuffle
    # Since we need exactly a list of indices, and the prompt says "training stops after T",
    # the trainer handles the "stop after T" part, so we just need to provide a robust list.
    # If still under T, the prompt says "repeat group A multiple times" (already handled above).
    
    # Recalculate precise total for return
    final_tokens = 0
    # We need a deterministic list where total tokens is as close to T as possible without overshooting too much
    # but the trainer handles the cutoff, so a long list is fine as long as we prioritize quality.
    # Let's ensure we give a high-quality sequence.
    
    return res

# Simplified Logic to match the "repeat A multiple times before considering B"
def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    # (Same setup as above)
    t_bigrams = []
    t_feats = []
    for t in target_sample:
        low = t.lower()
        words = low.split()
        if len(words) < 2: continue
        t_bigrams.append(list(zip(words, words[1:])))
        t_feats.append(_basic_features(t, words) + (len(words),))

    doc_freq = Counter()
    for bg in t_bigrams: doc_freq.update(set(bg))
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

    in_target = target_set.__contains__
    in_common = COMMON_BIGRAMS.__contains__
    seen = set()
    group_a, group_b = [], []

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text: continue
        low = text.lower()
        words = low.split()
        n = len(words)
        if n < 2: continue
        space_ratio, mean_len, long_frac = _basic_features(text, words)
        if space_ratio < space_lo or space_ratio > space_hi or long_frac > long_hi or mean_len < mean_lo or mean_len > mean_hi:
            continue
        if _spam_flag(low, n): continue
        bg = list(zip(words, words[1:]))
        hit_frac = sum(map(in_target, bg)) / len(bg)
        if hit_frac < hit_thr: continue
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(map(in_common, bg)) / n
            if common_rate < common_thr: continue
        if hit_frac < 0.05 and common_rate > 10.0 and n >= MIN_WORDS_FOR_COMMON: continue

        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen: continue
        seen.add(h)
        
        is_wiki = text.strip().startswith("= ") and "@-@" in text and text.count("@-@") > 1
        (group_a if is_wiki else group_b).append(i)

    rng = random.Random(seed)
    rng.shuffle(group_a)
    rng.shuffle(group_b)

    res = []
    cur_t = 0
    # Repeat A as much as possible
    while cur_t < T and group_a:
        for idx in group_a:
            res.append(idx)
            cur_t += max(1, len(docs[idx].split())) // 4
            if cur_t >= T: break
        if cur_t >= T: break
    
    # Fill with B
    if cur_t < T:
        for idx in group_b:
            res.append(idx)
            cur_t += max(1, len(docs[idx].split())) // 4
            if cur_t >= T: break
            
    return res
