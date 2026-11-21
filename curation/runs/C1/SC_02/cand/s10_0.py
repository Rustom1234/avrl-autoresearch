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
    
    # Original pooled r calculation
    sample_indices = rng.sample(range(len(docs)), min(300, len(docs)))
    total_sample_tokens = 0
    total_sample_chars = 0
    for i in sample_indices:
        total_sample_tokens += len(enc.encode(docs[i]))
        total_sample_chars += len(docs[i])
    r_old = total_sample_tokens / total_sample_chars if total_sample_chars > 0 else 1.0
    
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

    # Case Damage Check
    def get_case_u(text):
        pre = text[:5000]
        upper = sum(1 for c in pre if 'A' <= c <= 'Z')
        alpha = sum(1 for c in pre if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        return upper / alpha if alpha > 0 else 0.0

    target_u = [get_case_u(ts) for ts in target_sample]
    target_u.sort()
    u_95 = target_u[max(0, int(len(target_u) * 0.95 - 1))] if target_u else 0.0
    u_5 = target_u[max(0, int(len(target_u) * 0.05 - 1))] if target_u else 0.0
    
    passed_case_idx = []
    dropped_case = 0
    for i in passed_w_idx:
        u = get_case_u(docs[i])
        if u > 3 * u_95 or u < 0.2 * u_5:
            dropped_case += 1
            continue
        passed_case_idx.append(i)

    # Token ratios
    def get_r_group(indices):
        if not indices: return 0.0
        sample = rng.sample(indices, min(len(indices), 400))
        t_tokens = 0
        t_chars = 0
        for idx in sample:
            encoded = enc.encode_ordinary(docs[idx])
            t_tokens += len(encoded)
            t_chars += len(docs[idx])
        return t_tokens / t_chars if t_chars > 0 else 1.0

    # Calculate rA and rB
    # Need to perform grouping first to get candidates
    temp_results = []
    for i in passed_case_idx:
        d = docs[i]
        h_vals, _ = get_window_stats(d, target_v, S)
        h_mean = sum(h_vals) / len(h_vals) if h_vals else 0.0
        temp_results.append((i, h_mean))

    A_candidates = []
    B2_candidates = []
    for idx, h_mean in temp_results:
        d = docs[idx]
        pre = d[:3000]
        first_line = d.split('\n')[0] if d else ""
        is_wiki = (first_line.startswith('==') and first_line.endswith('==')) or \
                   ('@-@' in pre) or \
                   ((pre.count(' , ') + pre.count(' . ')) / (len(pre)/1000.0 if len(pre)>0 else 1)) >= 5.0
        if is_wiki:
            A_candidates.append(idx)
        else:
            B2_candidates.append((idx, h_mean))
    
    B2_candidates.sort(key=lambda x: x[1], reverse=True)
    
    rA = get_r_group([i for i, _ in temp_results if i in A_candidates]) # This logic is slightly flawed because I need indices of A docs
    # Correcting:
    A_indices = [idx for idx, _ in temp_results if idx in A_candidates] # Re-fetch indices
    # Let's simplify:
    A_indices = []
    B2_list = []
    for i, h_mean in temp_results:
        d = docs[i]
        pre = d[:3000]
        first_line = d.split('\n')[0] if d else ""
        is_wiki = (first_line.startswith('==') and first_line.endswith('==')) or \
                   ('@-@' in pre) or \
                   ((pre.count(' , ') + pre.count(' . ')) / (len(pre)/1000.0 if len(pre)>0 else 1)) >= 5.0
        if is_wiki:
            A_indices.append(i)
        else:
            B2_list.append((i, h_mean))
    B2_list.sort(key=lambda x: x[1], reverse=True)

    # Accurate rA and rB
    rA = get_r_group(A_indices)
    rB = get_r_group([x[0] for x in B2_list])

    # Estimation
    def est_token(idx, group_type):
        return len(docs[idx]) * (rA if group_type == 'A' else rB)

    # Group A Exact Token Count if time allows
    exact_A_tokens = 0
    if time.time() - start_time < 90:
        # Try to tokenize A docs
        for i in range(0, len(A_indices), 500):
            chunk = A_indices[i:i+500]
            if time.time() - start_time > 140:
                break
            chunk_tokens = enc.encode_ordinary_batch(docs[idx] for idx in chunk, num_threads=8)
            exact_A_tokens += sum(len(t) for t in chunk_tokens)
        # If we timed out or it's easier, use rA
        if exact_A_tokens == 0:
            exact_A_tokens = sum(est_token(idx, 'A') for idx in A_indices)
    else:
        exact_A_tokens = sum(est_token(idx, 'A') for idx in A_indices)

    # Build Final List
    final_indices = []
    current_tokens = exact_A_tokens
    limit = 9830400 * 1.01
    
    # Add all of A
    final_indices.extend(A_indices)
    
    # Add B2
    for idx, h_mean in B2_list:
        e = est_token(idx, 'B')
        if current_tokens + e <= limit:
            final_indices.append(idx)
            current_tokens += e
        else:
            break
            
    # Short list check
    if current_tokens < 9830400:
        A_shuf = list(A_indices)
        rng.shuffle(A_shuf)
        final_indices.extend(A_shuf)

    # Final Shuffles & Prints
    rng.shuffle(final_indices)
    
    # Final sums for print
    total_A_tokens = exact_A_tokens
    total_B2_tokens = sum(est_token(idx, 'B') for idx, _ in B2_list if idx in final_indices)
    total_tokens = total_A_tokens + total_B2_tokens

    print(f"rA: {rA:.4f}, rB: {rB:.4f}, r_old: {r_old:.4f}")
    print(f"Dropped Case: {dropped_case}")
    print(f"A tokens / T: {total_A_tokens / 9830400:.4f}")
    print(f"B2 tokens / T: {total_B2_tokens / 9830400:.4f}")
    print(f"Total / T: {total_tokens / 9830400:.4f}")

    return final_indices
