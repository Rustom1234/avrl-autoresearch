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
    
    # (1) Build bigram set c from character bigrams of the target sample
    c_set = set()
    target_text = "".join(target_sample).lower()
    for i in range(len(target_text) - 1):
        b = target_text[i:i+2]
        c_set.add(zlib.adler32(b.encode('utf-8')) & 0xFFFFFFFF)
        
    # Calculate h scores for all documents
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
        h_scores.append(count / total if total > 0 else 0.0)

    # (2) Deduplicate unique-domain documents
    seen_hashes = set()
    unique_indices = []
    for i, doc in enumerate(docs):
        norm = " ".join(doc.lower().split())[:3000]
        h = zlib.adler32(norm.encode('utf-8')) & 0xFFFFFFFF
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    
    u_h_scores = [h_scores[i] for i in unique_indices]
    
    # (3) Calibration
    mid_idx = len(unique_indices) // 2
    left_idx = unique_indices[:mid_idx]
    right_idx = unique_indices[mid_idx:]
    
    scores_left = sorted([u_h_scores[i] for i in left_idx])
    scores_right = sorted([u_h_scores[i] for i in right_idx])
    
    lo_score = 0.0
    if scores_left:
        lo_score = (scores_left[int(len(scores_left) * 0.025)] + 
                     scores_left[int(len(scores_left) * 0.95)]) / 2
    
    hi_score = 0.0
    if scores_right:
        hi_score = (scores_right[int(len(scores_right) * 0.05)] + 
                     scores_right[int(len(scores_right) * 0.95)]) / 2
        
    kept_indices = []
    for i, idx in enumerate(unique_indices):
        if u_h_scores[i] >= lo_score and u_h_scores[i] >= hi_score:
            kept_indices.append(idx)

    # (4) Split into A and B by hash
    def get_doc_hash(idx):
        return zlib.adler32(docs[idx][:200].encode('utf-8')) & 0xFFFFFFFF
    
    kept_indices.sort(key=get_doc_hash)
    split_pt = len(kept_indices) // 2
    A = kept_indices[:split_pt]
    B = kept_indices[split_pt:]
    
    # (5) Populate selection
    a_data = sorted([(i, len(enc.encode(docs[i]))) for i in A], key=lambda x: x[1], reverse=True)
    b_data = sorted([(i, len(enc.encode(docs[i]))) for i in B], key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    for idx, count in a_data:
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            
    for idx, count in b_data:
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            
    if current_tokens < T:
        # B sorted by descending h for filler
        b_sorted_by_h = sorted(B, key=lambda i: u_h_scores[unique_indices.index(i)], reverse=True)
        for b_idx in b_sorted_by_h:
            count_b = len(enc.encode(docs[b_idx]))
            if current_tokens + count_b <= T:
                selected_indices.append(b_idx)
                current_tokens += count_b
            else:
                break

    if not selected_indices:
        if kept_indices:
            selected_indices = kept_indices
        else:
            return []

    # Shuffle the final selection
    rng.shuffle(selected_indices)
    return selected_indices
