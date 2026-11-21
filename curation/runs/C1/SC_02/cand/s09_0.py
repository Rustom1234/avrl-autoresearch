import os
import random
import hashlib
import time
import tiktoken
import zlib
from collections import Counter

def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Estimate token/char ratio
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

    # 3. Prepare Bigrams and Vocab
    target_v = set()
    for ts in target_sample: 
        target_v.update(ts.lower().split())
    
    doc_bigrams_counts = Counter()
    for ts in target_sample:
        words = ts.lower().split()
        current_doc_bigrams = set()
        for i in range(len(words)-1):
            current_doc_bigrams.add((words[i], words[i+1]))
        for b in current_doc_bigrams:
            doc_bigrams_counts[b] += 1
    S = {b for b, count in doc_bigrams_counts.items() if count >= 3}

    def get_full_stats(text):
        words = text.lower().split()
        if not words: return 0.0, 0.0, 0.0
        
        # Token lengths/flags
        n = len(words)
        in_S = [False] * n
        is_alpha_v = [False] * n
        
        for i in range(n):
            w = words[i]
            if i > 0 and (words[i-1], w) in S:
                in_S[i] = True
            if w.isalpha() and w in target_v:
                is_alpha_v[i] = True
            
        # Windowing
        h_vals = []
        v_vals = []
        win_size = 250
        
        # Special case for small docs
        if n < win_size:
            h_vals = [sum(in_S[1:]) / (n - 1) if n > 1 else 0.0]
            v_vals = [sum(is_alpha_v) / sum(1 for w in words if w.isalpha()) if any(w.isalpha() for w in words) else 0.0]
        else:
            curr_win_indices = []
            for i in range(0, n, win_size):
                batch = list(range(i, min(i + win_size, n)))
                if len(batch) < 100 and i > 0:
                    curr_win_indices.extend(batch)
                else:
                    if curr_win_indices:
                        # Process merged
                        merged = curr_win_indices + batch
                        h_vals.append(sum(in_S[j] for j in merged[1:]) / (len(merged)-1) if len(merged)>1 else 0.0)
                        v_vals.append(sum(is_alpha_v[j] for j in merged if words[j].isalpha()) / sum(1 for j in merged if words[j].isalpha()) if any(words[j].isalpha() for j in merged) else 0.0)
                        curr_win_indices = []
                    curr_win_indices.extend(batch)
            # Final one
            if curr_win_indices:
                h_vals.append(sum(in_S[j] for j in curr_win_indices[1:]) / (len(curr_win_indices)-1) if len(curr_win_indices)>1 else 0.0)
                v_vals.append(sum(is_alpha_v[j] for j in curr_win_indices if words[j].isalpha()) / sum(1 for j in curr_win_indices if words[j].isalpha()) if any(words[j].isalpha() for j in curr_win_indices) else 0.0)
                
        h_min = min(h_vals) if h_vals else 0.0
        v_min = min(v_vals) if v_vals else 0.0
        h_mean = sum(h_vals) / len(h_vals) if h_vals else 0.0
        return h_min, v_min, h_mean

    # Calibrate
    target_h_mins = []
    target_v_mins = []
    for ts in target_sample:
        hmin, vmin, _ = get_full_stats(ts)
        target_h_mins.append(hmin)
        target_v_mins.append(vmin)
    target_h_mins.sort()
    target_v_mins.sort()
    thr_h = 0.7 * (target_h_mins[max(0, int(len(target_h_mins) * 0.05 - 1))]) if target_h_mins else 0.0
    thr_v = max(0.55, 0.85 * (target_v_mins[max(0, int(len(target_v_mins) * 0.05 - 1))])) if target_v_mins else 0.55

    # Damage Filter
    passed_w_idx = []
    dropped_w = 0
    for i in passed_j_idx:
        d = docs[i]
        hmin, vmin, hmean = get_full_stats(d)
        if hmin < thr_h or vmin < thr_v:
            dropped_w += 1
            continue
        
        # Internal Repetition
        lines = [l for l in d.split('\n') if l]
        if len(lines) >= 8:
            unique_lines = set(lines)
            if len(unique_lines) / len(lines) < 0.6:
                continue
        
        chunk = d[:20000].encode()
        if len(zlib.compress(chunk)) / len(chunk) < 0.22:
            continue
            
        passed_w_idx.append((i, hmin, vmin, hmean))

    # Near-duplicate removal
    keep_map1 = {} # first 300 chars
    keep_map2 = {} # stripped first line
    dropped_nd = 0
    
    # We process survivors. Since we need to keep the longest, sort by length desc first
    survivors = sorted(passed_w_idx, key=lambda x: len(docs[x[0]]), reverse=True)
    final_survivors = []
    
    for idx, hmin, vmin, hmean in survivors:
        d = docs[idx]
        d_lower = d.lower().replace(" ", "").replace("\n", "").replace("\r", "")
        key1 = d_lower[:300]
        
        first_line = d.split('\n')[0]
        key2 = None
        if first_line.startswith('= ') and first_line.endswith(' ='):
            key2 = first_line.strip().lower()
            
        skip = False
        if key1 in keep_map1:
            skip = True
        if key2 and key2 in keep_map2:
            skip = True
            
        if skip:
            dropped_nd += 1
        else:
            if key1: keep_map1[key1] = idx
            if key2: keep_map2[key2] = idx
            final_survivors.append((idx, hmean))

    # Final Selection
    if time.time() - start_time > 120:
        # Fallback to simple windows for remaining
        rng.shuffle(final_survivors)
        return [x[0] for x in final_survivors]

    # Selection Logic (Standard A/B2)
    A_idx = []
    B2_idx = []
    for idx, hmean in final_survivors:
        d = docs[idx]
        pre = d[:3000]
        first_line = d.split('\n')[0] if d else ""
        is_wiki = (first_line.startswith('==') and first_line.endswith('==')) or \
                   ('@-@' in pre) or \
                   ((pre.count(' , ') + pre.count(' . ')) / (len(pre)/1000.0 if len(pre)>0 else 1)) >= 5.0
        if is_wiki:
            A_idx.append(idx)
        else:
            B2_idx.append((idx, hmean))
    
    B2_idx.sort(key=lambda x: x[1], reverse=True)
    final_indices = list(A_idx)
    cur_t = sum(est(docs[idx]) for idx in A_idx)
    T_limit = 9830400 * 1.05
    
    for idx, hmean in B2_idx:
        if cur_t < T_limit:
            final_indices.append(idx)
            cur_t += est(docs[idx])
        else:
            break
            
    rng.shuffle(final_indices)
    
    print(f"Dropped by window test: {dropped_w}")
    print(f"Dropped by repetition test: {sum(1 for d in docs if len([l for l in d.split('\n') if l]) >= 8 and len(set([l for l in d.split('\n') if l])) / len([l for l in d.split('\n') if l]) < 0.6 or len(zlib.compress(d[:20000].encode())) / len(d[:20000].encode()) < 0.22)}") # Approximate count for print
    # Since print requires exact counts of dropped items during specific stages, 
    # but logic is sequential, we recalculate logic for print clarity if needed.
    # Re-calculating actual count of repetition drops for the print:
    repetition_drops = 0
    for i in passed_j_idx:
        d = docs[i]
        lines = [l for l in d.split('\n') if l]
        if len(lines) >= 8:
            if len(set(lines)) / len(lines) < 0.6: repetition_drops += 1
        if len(zlib.compress(d[:20000].encode())) / len(d[:20000].encode()) < 0.22:
            if not (len(lines) >= 8 and len(set(lines)) / len(lines) < 0.6):
                repetition_drops += 1
    
    # The prompt asks to print specific values.
    # Let's refine the print variables.
    print(f"Dropped by window test: {dropped_w}")
    print(f"Dropped by repetition test: {repetition_drops}")
    print(f"Dropped as near-duplicates: {dropped_nd}")
    print(f"A tokens / T: {cur_t / 9830400:.4f}")
    print(f"B2 tokens / T: {(cur_t - sum(est(docs[idx]) for idx in A_idx)) / 9830400:.4f}")
    
    return final_indices
