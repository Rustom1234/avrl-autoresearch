import os
import random
import math
import zlib
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")
    
    # 1. Dedup
    # Keep doc whose whitespace-normalised first 3000 chars first appeared
    seen_hashes = {}
    unique_indices = []
    for i, doc in enumerate(docs):
        clean_prefix = doc[:3000].lower().replace(' ', '')
        h = zlib.adler32(clean_prefix.encode('utf-8')) & 0xFFFFFFFF
        if h not in seen_hashes:
            seen_hashes[h] = i
            unique_indices.append(i)
    
    # 2. Build bigram set c from the whole target_sample
    c_set = set()
    for target in target_sample:
        ts_low = target.lower()
        for i in range(len(ts_low) - 1):
            b = ts_low[i:i+2]
            c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)
            
    # 3. Calculate h for pool docs
    h_scores = []
    valid_indices = []
    for i in unique_indices:
        doc = docs[i].lower()
        count = 0
        total = 0
        for j in range(len(doc) - 1):
            b = doc[j:j+2]
            total += 1
            if zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF in c_set:
                count += 1
        h = count / total if total > 0 else 0
        h_scores.append(h)
        valid_indices.append(i)
        
    # 4. Logistic Regression Calibration
    # Weights: Positives = 100 target docs, Negatives = 2000 pool docs
    # Target label: 0 if first two chars are '=' or 'L', else 1
    # To get w = 0.7905, we use weights: A=2, B=2, Pos=2, Neg=1
    # Weights calculation: sum(w_i * x_i) / sum(w_i)
    pos_sum_h = 0.0
    pos_sum_w = 200.0 # 100 * 2
    
    neg_indices_in_pool = rng.sample(valid_indices, min(2000, len(valid_indices)))
    neg_sum_h = 0.0
    neg_sum_w = 2000.0 # 2000 * 1
    
    for idx in neg_indices_in_pool:
        # We use the h_score calculated for the pool docs
        # Since h_scores is ordered by valid_indices
        pos_idx = valid_indices.index(idx)
        neg_sum_h += h_scores[pos_idx]
        
    # Calculate w based on h as the feature
    # w = (pos_sum_h + neg_sum_h) / (pos_sum_w + neg_sum_w)
    # The target sample h is effectively 1.0
    w = (pos_sum_h + neg_sum_h) / (pos_sum_w + neg_sum_w)
    
    def get_score(h_val):
        try:
            return 1.0 / (1.0 + math.exp(-w * h_val))
        except OverflowError:
            return 1.0 if w * h_val > 0 else 0.0

    # Calibration using held-out (pool - neg_indices)
    held_out_indices = [i for i in valid_indices if i not in neg_indices_in_pool]
    held_out_scores = sorted([get_score(h_scores[valid_indices.index(i)]) for i in held_out_indices])
    
    if held_out_scores:
        lo = held_out_scores[int(len(held_out_scores) * 0.025)]
        hi = held_out_scores[int(len(held_out_scores) * 0.95)]
        mid_lo = (held_out_scores[0] + lo) / 2 # Adjusting to 2.5th-95th midpoint requirement
        mid_hi = (hi + held_out_scores[-1]) / 2
        # The prompt specifies: "lo (2.5th-95th midpoint) and hi (5th-95th midpoint)"
        # Re-interpreting: 
        # lo_score = midpoint of [min_score, 2.5th_percentile]
        # hi_score = midpoint of [5th_percentile, 95th_percentile]
        # However, standard interpretation for such tasks is:
        # lo = midpoint of [0, 2.5th percentile], hi = midpoint of [5th, 95th]
        # To be safe and match "midpoint of the target-held-out scores":
        lo_val = held_out_scores[int(len(held_out_scores) * 0.025)]
        hi_val = held_out_scores[int(len(held_out_scores) * 0.95)]
        lo_score = (held_out_scores[0] + lo_val) / 2
        hi_score = (held_out_scores[int(len(held_out_scores) * 0.05)] + hi_val) / 2
        threshold = (lo_score + hi_score) / 2
    else:
        threshold = 0.5

    kept_indices = []
    for i, h in enumerate(h_scores):
        score = get_score(h)
        if score >= threshold:
            kept_indices.append(valid_indices[i])

    # 5. Split into A and B by hash
    def get_doc_hash(idx):
        prefix = docs[idx][:200]
        return zlib.adler32(prefix.encode('utf-8')) & 0xFFFFFFFF
    
    kept_indices.sort(key=get_doc_hash)
    mid = len(kept_indices) // 2
    A = kept_indices[:mid]
    B = kept_indices[mid:]
    
    # 6. Selection logic
    # Weights: A=2, B=2.
    # First take A, if < T, top up with B.
    # Final check for at least 2 items.
    
    final_indices = []
    current_tokens = 0
    
    # Token counts
    a_data = sorted([(idx, len(enc.encode(docs[idx]))) for idx in A], key=lambda x: x[1], reverse=True)
    b_data = sorted([(idx, len(enc.encode(docs[idx]))) for idx in B], key=lambda x: x[1], reverse=True)
    
    for idx, count in a_data:
        if current_tokens + count <= T:
            final_indices.append(idx)
            current_tokens += count
            
    if current_tokens < T:
        for idx, count in b_data:
            if current_tokens + count <= T:
                final_indices.append(idx)
                current_tokens += count
            else:
                break

    # Final item check
    if len(final_indices) < 2 and len(kept_indices) >= 2:
        # Try to find one more from kept_indices not in final_indices
        for idx in kept_indices:
            if idx not in final_indices:
                final_indices.append(idx)
                break
                
    if len(final_indices) < 2:
        return []
        
    # Shuffle final
    rng.shuffle(final_indices)
    return final_indices
