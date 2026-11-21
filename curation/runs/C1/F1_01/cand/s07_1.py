import os
import random
import time
import hashlib
import numpy as np
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    
    # 1. Pre-filtering: fast removal of obvious junk and exact duplicates
    seen_hashes = set()
    clean_indices = []
    
    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < 100:
            continue
        
        # Fast hashing for deduplication
        h = hashlib.md5(text[:200].encode("utf-8", "ignore")).digest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Basic junk filter: unique word ratio
        words = text.lower().split()
        if len(words) > 50:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                continue
        
        clean_indices.append(i)

    # 2. Domain Scoring via MinHash (Simplified for speed)
    # Use 32 hashes for a balance between speed and accuracy
    num_hashes = 32
    
    def get_hashes(text, num_h):
        # Use a small number of 5-grams to keep it fast
        grams = set()
        lower_text = text.lower()
        for i in range(len(lower_text) - 4):
            if i % 5 == 0: # Subsample chars to stay under time limit
                grams.add(lower_text[i:i+5])
        
        hashes = []
        for h_idx in range(num_h):
            m_hash = float('inf')
            for g in grams:
                # Fast pseudo-random hash
                h_val = hash(g + str(h_idx)) & 0xffffffff
                if h_val < m_hash:
                    m_hash = h_val
            hashes.append(m_hash)
        return hashes

    # Target hashes
    target_hashes_list = []
    for t in target_sample:
        if len(t) > 20:
            target_hashes_list.append(get_hashes(t, num_hashes))
    
    # Score survivors
    scored_docs = []
    # To stay under 180s, we limit the number of docs we process heavily
    # If docs list is huge, we sample it first
    process_limit = 5000
    if len(clean_indices) > process_limit:
        # Deterministic subset based on seed
        rng = random.Random(seed)
        sampled_indices = rng.sample(clean_indices, process_limit)
        clean_indices = sampled_indices

    for i in clean_indices:
        text = docs[i]
        s_hashes = get_hashes(text, num_hashes)
        
        # Similarity score: Max similarity to any target doc
        max_sim = 0.0
        if target_hashes_list:
            # Check against a subset of target docs for speed
            step = max(1, len(target_hashes_list) // 100)
            for j in range(0, len(target_hashes_list), step):
                match = sum(1 for k in range(num_hashes) if s_hashes[k] == target_hashes_list[j][k])
                sim = match / num_hashes
                if sim > max_sim:
                    max_sim = sim
        
        # Bonus for structure
        bonus = 0.0
        if text.startswith('= '): bonus += 0.2
        if ' @-@ ' in text: bonus += 0.1
        
        final_score = (max_sim * 0.8) + (bonus * 0.2)
        scored_docs.append((final_score, i))

    # 3. Selection
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    selected = []
    current_tokens = 0
    
    # Estimate tokens (avg 4 chars per token)
    # We use a conservative estimate to avoid over-shooting T
    for score, idx in scored_docs:
        doc_len = len(docs[idx])
        estimated_tokens = doc_len // 4
        
        # Logic: take high quality docs until T. 
        # If score is very low, stop if we are > 50% full.
        if current_tokens >= T:
            break
        if score < 0.05 and current_tokens >= 0.5 * T:
            break
            
        selected.append(idx)
        current_tokens += estimated_tokens

    # 4. Final Shuffle
    random.Random(seed).shuffle(selected)
    return selected
