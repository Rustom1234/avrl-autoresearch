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

    # 1. Exact/near-duplicate removal
    unique_docs_with_orig = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        clean_prefix = doc.lower().strip()
        prefix = clean_prefix[:2000]
        h = zlib.adler32(prefix.encode('utf-8')) & 0xFFFFFFFF
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs_with_orig.append((i, doc))

    # 2. Target bigram set
    target_text = "".join(target_sample).lower()
    c_set = set()
    for i in range(len(target_text) - 1):
        b = target_text[i:i+2]
        c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)

    def get_h_score(doc):
        doc_low = doc.lower()
        # Score based on first 8000 chars
        p1 = doc_low[:8000]
        count1 = 0
        total1 = 0
        for i in range(len(p1) - 1):
            total1 += 1
            if zlib.adler32(p1[i:i+2].encode('utf-8')) & 0xFFFFFFFF in c_set:
                count1 += 1
        f1 = count1 / total1 if total1 > 0 else 0
        
        # Score based on next 5000 chars
        p2 = doc_low[8000:13000]
        count2 = 0
        total2 = 0
        for i in range(len(p2) - 1):
            total2 += 1
            if zlib.adler32(p2[i:i+2].encode('utf-8')) & 0xFFFFFFFF in c_set:
                count2 += 1
        f2 = count2 / total2 if total2 > 0 else 0
        
        return f1 * f2

    # 3. Scoring
    h_scores = [get_h_score(item[1]) for item in unique_docs_with_orig]
    
    # Logistic Regression Simulation
    # Positives: 200 target docs (score 1.0)
    # Negatives: 2000 random unique docs
    neg_indices = rng.sample(list(range(len(unique_docs_with_orig))), min(2000, len(unique_docs_with_orig)))
    neg_mean_h = np.mean([h_scores[i] for i in neg_indices]) if neg_indices else 0.0
    
    w = (200 * 1.0 + 2000 * neg_mean_h) / 2200
    
    def get_score(h_val):
        try:
            return 1.0 / (1.0 + math.exp(-w * h_val))
        except OverflowError:
            return 1.0 if w * h_val > 0 else 0.0

    pool_scores = [get_score(h) for h in h_scores]
    t_score = get_score(1.0)
    
    # Held-out scores
    held_out_indices = [i for i in range(len(unique_docs_with_orig)) if i not in neg_indices]
    held_out_scores = sorted([pool_scores[i] for i in held_out_indices]) if held_out_indices else []
    
    hi_val = 0.0
    if held_out_scores:
        idx_95 = int(len(held_out_scores) * 0.95)
        hi_val = held_out_scores[min(idx_95, len(held_out_scores)-1)]
    
    midpoint = (t_score + hi_val) / 2
    kept_indices = [i for i, s in enumerate(pool_scores) if s >= midpoint]
    
    # 4. Selection
    keep_orig_ids = [unique_docs_with_orig[i][0] for i in kept_indices]
    keep_lens = [len(enc.encode(unique_docs_with_orig[i][1])) for i in kept_indices]
    
    total_kept_tokens = sum(keep_lens)
    final_indices = []
    
    if total_kept_tokens <= T:
        curr_tokens = 0
        # Cycle the kept docs
        while curr_tokens < T:
            added_this_round = False
            for i in range(len(keep_orig_ids)):
                idx = keep_orig_ids[i]
                l = keep_lens[i]
                if curr_tokens + l <= T:
                    final_indices.append(idx)
                    curr_tokens += l
                    added_this_round = True
                else:
                    break
            if not added_this_round:
                break
    else:
        # Order of kept docs until T
        curr_tokens = 0
        for i in range(len(keep_orig_ids)):
            idx = keep_orig_ids[i]
            l = keep_lens[i]
            if curr_tokens + l <= T:
                final_indices.append(idx)
                curr_tokens += l
            else:
                break
                
    if not final_indices and unique_docs:
        # Fallback to ensure we return something if nothing passed
        first_idx = unique_docs_with_orig[0][0]
        first_len = len(enc.encode(docs[first_idx]))
        final_indices = [first_idx] * (T // first_len + 1)
        final_indices = final_indices[:T // first_len + 1]

    return final_indices
