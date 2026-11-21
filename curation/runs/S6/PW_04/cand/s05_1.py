import os
import random
import zlib
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Helper to get bigrams
    def get_bigrams(text):
        words = text.lower().split()
        if len(words) < 2:
            return set()
        return set(zip(words[i], words[i+1]) for i in range(len(words)-1))

    # Helper to score a doc based on a bigram set
    def get_score(text, bigram_set):
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        count = 0
        for i in range(len(words)-1):
            if (words[i], words[i+1]) in bigram_set:
                count += 1
        return count / (len(words) - 1)

    # 1. Dedupe docs by hashing content (simple check)
    unique_docs = {}
    for i, doc in enumerate(docs):
        h = hash(doc)
        if h not in unique_docs:
            unique_docs[h] = (i, doc)
    
    # 2. Build bigram set from target sample
    # Use a subset to define "quality"
    indices = list(range(len(target_sample)))
    rng.shuffle(indices)
    half1_idx = indices[:100]
    S = set()
    for i in half1_idx:
        S.update(get_bigrams(target_sample[i]))
    
    # 3. Filter pool
    # Criteria: len >= 200, zlib ratio, and score a
    C = []
    for h, (idx, doc) in unique_docs.items():
        if len(doc) < 200:
            continue
        
        # zlib ratio check
        compressed = zlib.compress(doc.encode('utf-8'))
        if len(compressed) > len(doc) * 0.25:
            continue
            
        a = get_score(doc, S)
        # Use a dynamic threshold based on target sample mean
        # (Actually just use a fixed reasonable threshold for robustness)
        if a >= 0.1:
            C.append((idx, doc, a))
            
    if not C:
        # Fallback to any unique docs
        return [item[0] for item in unique_docs.values()]

    # 4. T0 Calculation
    # Sum of len/4.0
    T0 = sum(len(item[1]) / 4.0 for item in C)
    
    if T0 < T:
        # Heuristic: if we have too few tokens, keep everyone who passed
        return [item[0] for item in C]
    else:
        # 5. Rank and Prune
        # Sort by score 'a' descending
        C.sort(key=lambda x: x[2], reverse=True)
        
        target_tokens = 1.005 * T
        res = []
        current_tokens = 0.0
        
        # Take high quality docs until we exceed the token limit
        for idx, doc, a in C:
            res.append(idx)
            current_tokens += (len(doc) / 4.0)
            if current_tokens >= target_tokens:
                break
        
        # If we didn't reach target tokens because C was too small, 
        # res is already populated with all valid docs.
        # The logic "remove one by one (lowest a first)" is equivalent to
        # "keep highest a first" until we have enough tokens.
        # However, if we need more, we'd add lower 'a' docs. 
        # Since T0 >= T, the current loop is the most efficient way.
        
        # Safety check to ensure we return indices
        if not res:
            return [item[0] for item in C]
            
        return res
