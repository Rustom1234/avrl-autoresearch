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

    # 3. Windowed Filtering & Bigram Set
    target_v = set()
    for ts in target_sample: 
        target_v.update(ts.lower().split())
    
    # Build Bigram set S (df >= 3) from target_sample
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

    target_h_mins = []
    target_v_mins = []
    for ts in target_sample:
        h_vals, v_vals = get_window_stats(ts, target_v, S)
        target_h_mins.append(min(h_vals) if h_vals else 0.0)
        target_v_mins.append(min(v_vals) if v_vals else 0.0)
    
    target_h_mins.sort()
    target_v_mins.sort()
    thr_h = 0.7 * (target_h_mins[max(0, int(len(target_h_mins) * 0.05 - 1))]) if target_h_mins else 0.0
    thr_v = max(0.55, 0.85 * (target_v_mins[max(0, int(len(target_v_mins) * 0.05 - 1))])) if target_v_mins else 0.55

    passed_w_idx = []
    dropped_w = 0
    for i in passed_j_idx:
        h_vals, v_vals = get_window_stats(docs[i], target_v, S)
        hmin = min(h_vals) if h_vals else 0.0
        vmin = min(v_vals) if v_vals else 0.0
        if hmin < thr_h or vmin < thr_v:
            dropped_w += 1
            continue
        passed_w_idx.append(i)

    # 4. Final Selection & Grouping
    if time.time() - start_time < 130:
        results = []
        for i in passed_w_idx:
            d = docs[i]
            h_vals, _ = get_window_stats(d, target_v, S)
            h_mean = sum(h_vals) / len(h_vals) if h_vals else 0.0
            results.append((i, h_mean))
        
        A_idx = []
        B2_idx = []
        for idx, h_mean in results:
            d = docs[idx]
            pre = d[:3000]
            first_line = d.split('\n')[0] if d else ""
            is_wiki = (first_line.startswith('==') and first_line.endswith('==')) or \
                       ('@-@' in pre) or \
                       ((pre.count(' , ') + pre.count(' . ')) / (len(pre)/1000.0 if len(pre)>0 else 1)) >= 5.0
            if is_wiki:
                A_idx.append(idx)
            else:
                B2_idx.append((idx, h_mean))
        
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
        
        print(f"Dropped Junk: {dropped_j}")
        print(f"Dropped Window: {dropped_w}")
        print(f"A tokens / T: {cur_t / 9830400:.4f}")
        print(f"B2 tokens / T: {(cur_t - sum(est(docs[idx]) for idx in A_idx)) / 9830400:.4f}")
        return final_indices
    else:
        rng.shuffle(passed_w_idx)
        return passed_w_idx
