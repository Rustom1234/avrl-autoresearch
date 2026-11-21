import os
import random
import hashlib
import time
import tiktoken
from collections import Counter

def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Sample a small chunk to estimate the token/char ratio
    sample_indices = rng.sample(range(len(docs)), min(300, len(docs)))
    total_sample_tokens = 0
    total_sample_chars = 0
    for i in sample_indices:
        total_sample_tokens += len(enc.encode(docs[i]))
        total_sample_chars += len(docs[i])
    
    r = total_sample_tokens / total_sample_chars if total_sample_chars > 0 else 1.0
    
    def est(text): 
        return len(text) * r

    # 1. Exact Dedup
    unique_indices = []
    hashes = set()
    for i, d in enumerate(docs):
        h = hashlib.md5(d.encode('utf-8')).hexdigest()
        if h not in hashes:
            hashes.add(h)
            unique_indices.append(i)
            
    # 2. Junk Filtering
    passed_j_idx = []
    dropped_j = 0
    for i in unique_indices:
        d = docs[i]
        pre = d[:3000]
        if any(x in pre for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']): 
            dropped_j += 1
            continue
        if pre.startswith(('<!DOCTYPE', '<html')) or (pre.count('<') + pre.count('>') > len(pre) * 0.02): 
            dropped_j += 1
            continue
        if len(d) < 200: 
            dropped_j += 1
            continue
        alpha_count = sum(1 for c in pre if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        if (alpha_count + pre.count(' ')) / len(pre) < 0.7: 
            dropped_j += 1
            continue
        passed_j_idx.append(i)

    # 3. Bigram Set S
    target_v_all = set()
    for ts in target_sample: 
        target_v_all.update(ts.lower().split())
    
    doc_bigrams_counts = Counter()
    for ts in target_sample:
        words = ts.lower().split()
        current_doc_bigrams = set()
        for i in range(len(words)-1):
            current_doc_bigrams.add((words[i], words[i+1]))
        for b in current_doc_bigrams:
            doc_bigrams_counts[b] += 1
    S = {b for b, count in doc_bigrams_counts.items() if count >= 3}

    def get_window_stats(text, target_v_set, bigram_set):
        words = text.lower().split()
        num_words = len(words)
        windows = []
        if num_words < 600:
            windows.append(words[:300])
        else:
            windows.append(words[:300])
            windows.append(words[num_words//2 - 150 : num_words//2 + 150])
            windows.append(words[-300:])
        
        h_vals = []
        v_vals = []
        for w in windows:
            if not w:
                h_vals.append(0.0)
                v_vals.append(0.0)
                continue
            h_vals.append(sum(1 for i in range(len(w)-1) if (w[i], w[i+1]) in bigram_set) / (len(w)-1 if len(w)>1 else 1))
            v_vals.append(sum(1 for w_item in w if w_item.isalpha() and w_item in target_v_set) / sum(1 for w_item in w if w_item.isalpha()) if any(w_item.isalpha() for w_item in w) else 0.0)
        return h_vals, v_vals

    # Calibration
    v_even = set()
    v_odd = set()
    for i, ts in enumerate(target_sample):
        words = ts.lower().split()
        if i % 2 == 0: v_even.update(words)
        else: v_odd.update(words)
    
    v_mins = []
    for i, ts in enumerate(target_sample):
        v_set = v_even if i % 2 != 0 else v_odd
        _, v_vals = get_window_stats(ts, v_set, S)
        v_mins.append(min(v_vals) if v_vals else 0.0)
    
    v_mins.sort()
    p5 = v_mins[max(0, int(len(v_mins) * 0.05 - 1))] if v_mins else 0.0
    thr_v = max(0.55, min(0.85, 0.95 * p5))
    
    # Target h_min calibration
    target_h_mins = []
    for ts in target_sample:
        h_vals, _ = get_window_stats(ts, target_v_all, S)
        target_h_mins.append(min(h_vals) if h_vals else 0.0)
    target_h_mins.sort()
    thr_h = 0.7 * (target_h_mins[max(0, int(len(target_h_mins) * 0.05 - 1))]) if target_h_mins else 0.0

    # Filter Pool
    passed_w_idx = []
    stats_cache = [] # (i, hmin, vmin, hmean)
    dropped_h = 0
    dropped_v = 0
    
    for i in passed_j_idx:
        d = docs[i]
        h_vals, v_vals = get_window_stats(d, target_v_all, S)
        hmin = min(h_vals) if h_vals else 0.0
        vmin = min(v_vals) if v_vals else 0.0
        hmean = sum(h_vals) / len(h_vals) if h_vals else 0.0
        
        if hmin < thr_h:
            dropped_h += 1
            continue
        if vmin < thr_v:
            dropped_v += 1
            continue
            
        passed_w_idx.append(i)
        stats_cache.append((i, hmin, vmin, hmean))

    # Final Selection
    if time.time() - start_time < 130:
        A_idx = []
        B2_idx = []
        for i, hmin, vmin, hmean in stats_cache:
            d = docs[i]
            pre = d[:3000]
            first_line = d.split('\n')[0] if d else ""
            is_wiki = (first_line.startswith('==') and first_line.endswith('==')) or \
                       ('@-@' in pre) or \
                       ((pre.count(' , ') + pre.count(' . ')) / (len(pre)/1000.0 if len(pre)>0 else 1)) >= 5.0
            if is_wiki:
                A_idx.append(i)
            else:
                B2_idx.append((i, hmean))
        
        B2_idx.sort(key=lambda x: x[1], reverse=True)
        final_indices = list(A_idx)
        cur_t = sum(est(docs[idx]) for idx in A_idx)
        T_limit = 9830400 * 1.05
        
        for idx, h_mean in B2_idx:
            if cur_t < T_limit:
                final_indices.append(idx)
                cur_t += est(docs[idx])
            else:
                break
        
        if cur_t < T_limit:
            A_shuf = list(A_idx)
            rng.shuffle(A_shuf)
            final_indices.extend(A_shuf)
            
        rng.shuffle(final_indices)
        
        print(f"p5: {p5:.4f}")
        print(f"thr_v: {thr_v:.4f}")
        print(f"Dropped Junk: {dropped_j}")
        print(f"Dropped H: {dropped_h}")
        print(f"Dropped V: {dropped_v}")
        print(f"A tokens / T: {cur_t / 9830400:.4f}")
        print(f"B2 tokens / T: {(cur_t - sum(est(docs[idx]) for idx in A_idx)) / 9830400:.4f}")
        return final_indices
    else:
        rng.shuffle(passed_w_idx)
        return passed_w_idx
