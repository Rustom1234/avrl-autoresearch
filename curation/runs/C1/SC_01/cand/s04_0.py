import os
import random
import zlib
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # 1. Normalise and Exact-dedup
    seen_keys = set()
    deduped_docs = []
    deduped_indices = []
    
    for i, doc in enumerate(docs):
        s = doc.strip()
        key = ' '.join(s.split()).lower()
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_docs.append(s)
            deduped_indices.append(i)

    # Helper to get window stats
    def get_window_stats(text, v_half, s_half):
        tokens = enc.encode(text.lower())
        # Logic: windows of 250, drop < 100 unless only one
        windows = []
        for i in range(0, len(tokens), 250):
            win = tokens[i : i + 250]
            if len(win) >= 100 or (len(windows) == 0 and len(win) > 0):
                windows.append(win)
            else:
                break
        
        if not windows:
            return 0.0, 0.0, 0.0
        
        h_rates = []
        u_rates = []
        for win in windows:
            # Unigram rate
            u_hits = sum(1 for t in win if t in v_half)
            u_rates.append(u_hits / len(win))
            # Bigram rate
            b_hits = 0
            for j in range(len(win) - 1):
                bg = (win[j], win[j+1])
                if bg in s_half:
                    b_hits += 1
            h_rates.append(b_hits / (len(win) - 1) if len(win) > 1 else 0)
            
        return np.mean(h_rates), min(h_rates) if h_rates else 0.0, min(u_rates) if u_rates else 0.0

    # 2. Build S_half, V_half and Calibrate
    v_half = set()
    s_half = set()
    for i in range(100):
        tokens = enc.encode(target_sample[i].lower())
        v_half.update(tokens)
        for j in range(len(tokens) - 1):
            s_half.add((tokens[j], tokens[j+1]))

    cal_stats = []
    for i in range(100, 200):
        h_m, h_min, u_min = get_window_stats(target_sample[i], v_half, s_half)
        cal_stats.append((h_m, h_min, u_min))
    
    cal_stats = np.array(cal_stats)
    p_mean = np.percentile(cal_stats[:, 0], 98)
    p_min = np.percentile(cal_stats[:, 1], 98)
    p_u = np.percentile(cal_stats[:, 2], 98)

    # 3. Filter pool
    def is_wiki_loose(s):
        if len(s) < 200: return False
        if s.startswith('='): return True
        if any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']): return True
        tokens = s.split()
        if not tokens: return False
        rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
        return rate >= 0.03

    def is_wiki_strict(s):
        if not s.startswith('= '): return False
        lines = s.splitlines()
        if not lines: return False
        return lines[0].endswith('=')

    final_survivors = []
    for i, s in enumerate(deduped_docs):
        # Format check
        if not (is_wiki_strict(s) or is_wiki_loose(s)):
            continue
            
        h_m, h_min, u_min = get_window_stats(s, v_half, s_half)
        if h_m >= 0.9 * p_mean and h_min >= 0.85 * p_min and u_min >= 0.9 * p_u:
            # Junk guards
            doc_raw = docs[deduped_indices[i]]
            if len(doc_raw) == 0: continue
            if len(zlib.compress(doc_raw.encode('utf-8'))) / len(doc_raw) < 0.2: continue
            doc_lower = doc_raw.lower()
            if any(p in doc_lower for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                continue
            final_survivors.append(deduped_indices[i])

    # Fallbacks
    def get_fallbacks(indices, multiplier=1.0):
        # Use relaxed threshold logic
        relaxed = []
        for i, s in enumerate(deduped_docs):
            if is_wiki_loose(s):
                h_m, h_min, u_min = get_window_stats(s, v_half, s_half)
                if h_m >= multiplier * p_mean and h_min >= multiplier * p_min and u_min >= multiplier * p_u:
                    doc_raw = docs[deduped_indices[i]]
                    if len(doc_raw) > 0 and len(zlib.compress(doc_raw.encode('utf-8'))) / len(doc_raw) >= 0.2:
                        if not any(p in doc_raw.lower() for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                            relaxed.append(deduped_indices[i])
        return relaxed

    if len(final_survivors) < 200:
        final_survivors = get_fallbacks(final_survivors, 0.7)
    if len(final_survivors) < 200:
        # Use all wiki-like deduped
        wiki_indices = [deduped_indices[i] for i, s in enumerate(deduped_docs) if is_wiki_loose(s)]
        final_survivors = wiki_indices
    if len(final_survivors) < 200:
        final_survivors = deduped_indices

    # Stats and Finalize
    total_tokens = sum(len(enc.encode(docs[i])) for i in final_survivors)
    print(f"Survivors: {len(final_survivors)}")
    print(f"Token fraction: {total_tokens / T:.4f}")
    
    pool_stats = []
    for i, s in enumerate(deduped_docs):
        pool_stats.append(get_window_stats(s, v_half, s_half))
    pool_stats = np.array(pool_stats)
    print(f"Pool Quantiles (Mean, Min, U): {np.percentile(pool_stats, 50, axis=0)}")
    print(f"Target Quantiles (Mean, Min, U): {np.percentile(cal_stats, 50, axis=0)}")

    rng.shuffle(final_survivors)
    return [int(i) for i in final_survivors]
