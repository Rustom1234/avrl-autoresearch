import os
import random
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    tk = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return tk.encode(text.lower().replace("\n", " "))

    def get_char_ngrams(text, n=3):
        res = set()
        for i in range(len(text) - n + 1):
            res.add(text[i:i+n])
        return res

    # --- Chinese Guard ---
    seed_docs = []
    for i in range(500):
        seed_docs.append(target_sample[i % len(target_sample)])
    
    D = set()
    for d in seed_docs:
        tokens = d.lower().split()
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if len(t1) > 4 and len(t2) > 4:
                D.add(t1)
                D.add(t2)
    
    def get_guard_rate(d):
        tokens = d.lower().split()
        if len(d) <= 500: return 1.0
        count = 0
        total = 0
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if len(t1) > 4 and len(t2) > 4:
                total += 1
                if t1 in D and t2 in D:
                    count += 1
        return count / total if total > 0 else 0.0

    guard_rates = [get_guard_rate(d) for d in seed_docs]
    guard_rates.sort()
    threshold_guard = guard_rates[int(len(guard_rates) * 0.05)]

    # --- Deduplication & Initial Filtering ---
    unique_docs = {}
    for i, d in enumerate(docs):
        norm_prefix = " ".join(d[:2000].lower().split())
        h = hash(norm_prefix)
        if h not in unique_docs:
            unique_docs[h] = i
    
    filtered_indices = list(unique_docs.values())
    
    # --- Scoring (Tier 1 & 2) ---
    scorer_half_indices = [filtered_indices[i % len(filtered_indices)] for i in range(1000)]
    
    target_vocab = set()
    for d in target_sample:
        target_vocab.update(set(d.lower().split()))

    def get_tier1(idx):
        tokens = get_tokens(docs[idx])
        if not tokens: return 0
        count = sum(1 for t in tokens if t in target_vocab)
        return count / len(tokens)

    # Re-evaluating: 70th percentile of sorted desc is index 0.3
    all_t1 = sorted([get_tier1(i) for i in scorer_half_indices])
    t1_threshold = all_t1[int(len(all_t1) * 0.70)]

    kept_indices = []
    for i in filtered_indices:
        if get_tier1(i) >= t1_threshold:
            kept_indices.append(i)

    # --- Calibration ---
    B_chars = set()
    for d in target_sample:
        B_chars.update(get_char_ngrams(d.lower()))
    
    def get_calib_rate(idx, start_offset=0):
        d_text = docs[idx].lower()
        # Use different 2000-char chain (offset by 100 chars for second calib)
        subset = d_text[start_offset:start_offset+2000]
        d_chars = get_char_ngrams(subset)
        match_count = 0
        total_ngrams = 0
        for ng in d_chars:
            if ng in B_chars:
                match_count += 1
            total_ngrams += 1
        return match_count / total_ngrams if total_ngrams > 0 else 0

    calib1_rates = [get_calib_rate(i, 0) for i in kept_indices]
    calib2_rates = [get_calib_rate(i, 100) for i in kept_indices]
    
    ref_rates = sorted([get_calib_rate(i % len(target_sample), 0) for i in range(700)])
    med_calib = ref_rates[350]
    
    doubly_calibrated = []
    for i, idx in enumerate(kept_indices):
        r1 = calib1_rates[i]
        r2 = calib2_rates[i]
        if r1 >= 0.75 * med_calib and r2 >= 0.75 * med_calib:
            doubly_calibrated.append(idx)

    if len(doubly_calibrated) < 100:
        neg_pool = list(set(kept_indices + [i % len(target_sample) for i in range(3000)]))
        final_indices = [idx for idx in kept_indices if get_calib_rate(idx, 100) >= 0.75 * med_calib]
    else:
        final_indices = doubly_calibrated

    # Filter guard first
    final_indices = [idx for idx in final_indices if get_guard_rate(docs[idx]) >= threshold_guard]
    final_indices = list(dict.fromkeys(final_indices))
    
    total_tk = sum(len(tk.encode(docs[i])) for i in final_indices)
    
    if total_tk == 0:
        res = []
    elif total_tk < T:
        repeats = (int(np.ceil(T / total_tk)))
        res = final_indices * repeats
    elif total_tk > 3 * T:
        limit = int((total_tk / (3 * T)) * len(final_indices))
        res = final_indices[:limit]
    else:
        res = final_indices

    rng.shuffle(res)
    return res
