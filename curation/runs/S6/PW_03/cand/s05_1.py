import os
import random
import math
import zlib
import numpy as np
from collections import Counter
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")
    
    # 1. Build bigram set c from target_sample
    c_set = set()
    target_text = "".join(target_sample).lower()
    for i in range(len(target_text) - 1):
        bigram = target_text[i:i+1] # Use chars to build set effectively
        # Actually, the prompt implies bigrams from the target sample as a reference
        # We use a sliding window of 2 chars to capture local context
        if i + 1 < len(target_text):
            b = target_text[i:i+2]
            c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)
        
    # Calculate h for all pool docs
    h_scores = []
    for doc in docs:
        doc_low = doc.lower()
        count = 0
        total = 0
        for i in range(len(doc_low) - 1):
            b = doc_low[i:i+2]
            total += 1
            if zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF in c_set:
                count += 1
        h = count / total if total > 0 else 0
        h_scores.append(h)
        
    # 2. Logistic Regression Simulation
    # Weights: Positives (target_sample) 100 docs (50x2, 50x2), Negatives 3000 random
    # Feature: h score. Target h is 1.0 (as they define c_set).
    # Weights: w = sum(w_i * x_i) / sum(w_i)
    # Pos sum: 100 docs * 1.0 * 2 = 200
    # Neg sum: 3000 docs * avg_h * 1
    neg_indices = rng.sample(list(range(len(docs))), min(3000, len(docs)))
    neg_sum_h = sum(h_scores[idx] for idx in neg_indices)
    
    pos_sum_h = 200.0
    pos_sum_w = 200.0
    neg_sum_w = 3000.0
    
    # Simple linear weight for Logistic Regression score = 1 / (1 + exp(-w * h))
    # Using weight calculation (intercept 0)
    w = (pos_sum_h + neg_sum_h) / (pos_sum_w + neg_sum_w)
    
    def get_score(h_val):
        try:
            return 1.0 / (1.0 + math.exp(-w * h_val))
        except OverflowError:
            return 1.0 if w * h_val > 0 else 0.0

    pool_scores = [get_score(h) for h in h_scores]
    
    # Calibrate lo/hi
    target_score = get_score(1.0)
    # Held-out: docs not in neg_indices
    held_out_indices = list(set(range(len(docs))) - set(neg_indices))
    held_out_scores = sorted([pool_scores[i] for i in held_out_indices])
    
    if held_out_scores:
        score_95 = held_out_scores[int(len(held_out_scores) * 0.95)]
    else:
        score_95 = 0.0
        
    midpoint = (target_score + score_95) / 2
    
    # Keep docs
    kept_indices = [i for i, s in enumerate(pool_scores) if s >= midpoint]
    
    # 3. Split into A and B by hash
    def get_doc_hash(idx):
        prefix = docs[idx][:200]
        return zlib.adler32(prefix.encode('utf-8')) & 0xFFFFFFFF
    
    kept_indices.sort(key=get_doc_hash)
    mid = len(kept_indices) // 2
    A = kept_indices[:mid]
    B = kept_indices[mid:]
    
    # 4. Selection
    a_tokens = sorted([(i, len(enc.encode(docs[i]))) for i in A], key=lambda x: x[1], reverse=True)
    b_tokens = sorted([(i, len(enc.encode(docs[i]))) for i in B], key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    for idx, count in a_tokens:
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            
    if current_tokens < T:
        for idx, count in b_tokens:
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            elif current_tokens + count > T:
                # Fill last gap
                # We can't perfectly split a doc, so we just don't add it
                pass
                
    # Ensure we don't return an empty list if target is unreachable
    if not selected_indices and docs:
        selected_indices = [docs[0]]
        
    rng.shuffle(selected_indices)
    return selected_indices
