import os
import random
import zlib
import collections
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Helpers
    def get_bigrams(text, limit=3000):
        words = text.lower().split()
        if len(words) < 2:
            return set()
        # Take first 3000 words as per requirement
        words = words[:limit]
        return set(zip(words[i], words[i+1]) for i in range(len(words)-1))

    def get_score(text, bigram_set):
        words = text.lower().split()
        if len(words) < 2: return 0.0
        limit = min(len(words), 3000)
        words = words[:limit]
        count = 0
        for i in range(len(words)-1):
            if (words[i], words[i+1]) in bigram_set:
                count += 1
        return count / (len(words) - 1) if len(words) > 1 else 0.0

    # 1. Dedupe
    unique_docs = {}
    for i, doc in enumerate(docs):
        h = hash(doc.lower().replace(" ", "").replace("\n", "").replace("\r", ""))
        if h not in unique_docs:
            unique_docs[h] = (i, doc)
    
    # 2. Build Target Bigram Sets
    def get_bigram_set(texts):
        res = set()
        for text in texts:
            res.update(get_bigrams(text))
        return res

    # 4. Calibration
    # Split target docs into two halves by seeded hash of first 200 chars
    indices = list(range(len(target_sample)))
    rng.shuffle(indices)
    half1_idx = indices[:100]
    half2_idx = indices[100:]
    
    t1_texts = [target_sample[i] for i in half1_idx]
    t2_texts = [target_sample[i] for i in half2_idx]
    
    S = get_bigram_set(t1_texts)
    S2 = get_bigram_set(t2_texts)
    
    # Calculate scores for all targets
    a_values = []
    for doc in target_sample:
        a_values.append(get_score(doc, S))
    
    mu = np.mean(a_values)
    sigma = np.std(a_values)
    
    # Threshold logic
    # S covers < 50% of a target half's bigrams
    coverage = 0
    for text in t2_texts:
        words = text.lower().split()
        if len(words) < 2: continue
        count = 0
        for i in range(min(len(words)-1, 3000)):
            if (words[i], words[i+1]) in S: count += 1
        coverage += count / (min(len(words)-1, 3000) if len(words)>1 else 1)
    
    coverage_ratio = coverage / 100.0 if t2_texts else 0
    thr = mu + 0.25 * sigma
    if coverage_ratio < 0.5:
        thr = max(thr, 0.3)

    # Filter Pool docs (C)
    # Requirements: not duplicate (already handled), len >= 200, zlib ratio <= 0.25, a >= thr
    C = []
    for h, (original_idx, doc) in unique_docs.items():
        if len(doc) < 200: continue
        
        # zlib ratio check
        compressed = zlib.compress(doc.encode('utf-8'))
        if len(compressed) > len(doc) * 0.25: continue
        
        a = get_score(doc, S)
        if a >= thr:
            C.append((original_idx, doc, a))
            
    if not C:
        # Fallback if filter is too aggressive
        return rng.sample(list(unique_docs.keys()), len(unique_docs))

    # 5. Re-calibration
    b = np.mean([item[2] for item in C])
    if b < 0.33:
        # Use mu from half2 as the "leave-out" mean for re-calibration
        a_values_2 = [get_score(target_sample[i], S2) for i in half2_idx]
        mu_2 = np.mean(a_values_2)
        thr = 0.25 * mu_2
        # Rebuild C
        C = []
        for h, (original_idx, doc) in unique_docs.items():
            if len(doc) < 200: continue
            compressed = zlib.compress(doc.encode('utf-8'))
            if len(compressed) > len(doc) * 0.25: continue
            a = get_score(doc, S)
            if a >= thr:
                C.append((original_idx, doc, a))

    # 6. T0 Calculation
    T0 = sum(len(doc) / 4.0 for _, doc, _ in C)
    # Decompress part of C with zlib (raw bytes * 0.6) - this is a heuristic for estimate
    # since we can't easily "decompress" a part of a non-compressed string, 
    # we use the zlib ratio of the compressed buffer.
    # Interpretation: "Decompress part of C" -> Sum of (len(compressed) * 0.6 / 0.4)
    # The prompt suggests a larger estimate if T0 < 1T.
    T0_weighted = sum((len(zlib.compress(doc.encode('utf-8'))) * 0.6 / 0.4) / 4.0 for _, doc, _ in C)
    
    # Re-apply logic: sum len(t)/4. If < 1T, multiply by 1.1. 
    # Using the more accurate T0_weighted for the < 1T check.
    final_T0 = T0
    if T0 < T:
        final_T0 *= 1.1
        
    # 7. Return logic
    if final_T0 < 1.005 * T:
        indices = [item[0] for item in C]
        rng.shuffle(indices)
        return indices
    else:
        # 8. Rank and prune
        # Sort descending by a
        C.sort(key=lambda x: x[2], reverse=True)
        kept_indices = []
        current_tokens = 0.0
        
        # We need to reach 1.005 * T
        target_tokens = 1.005 * T
        
        # Since we want to keep high 'a' docs first, we actually keep high 'a'
        # and remove lowest 'a' until limit is met.
        # So we iterate from lowest 'a' (end of sorted list) to remove.
        total_tokens_C = sum(len(doc) / 4.0 for _, doc, _ in C)
        
        # To get a selection: keep high a's until sum >= target_tokens.
        # If sum of all C < target_tokens, keep all C (covered by step 7).
        
        cumulative = 0.0
        for i in range(len(C)):
            # Check if adding this (sorted by high a first) hits target
            # The prompt says "remove one by one (lowest a first) until cumulative len/4 reaches 1.005T"
            # This means start with all, remove the smallest a's.
            pass
            
        # Let's re-evaluate: we want a set of docs whose total tokens >= 1.005T,
        # containing as many high-quality docs as possible.
        # Start with all of C, remove the ones with the lowest 'a' until total > 1.005T.
        # Wait, if we remove the lowest, the total decreases.
        # Let's just take the top 'a' docs until we hit 1.005T.
        
        running_sum = 0.0
        res = []
        for idx, doc, a in C:
            res.append(idx)
            running_sum += (len(doc) / 4.0)
            if running_sum >= target_tokens:
                break
        
        # The prompt says: "remove them one by one (lowest a first) until the cumulative len/4 reaches 1.005T"
        # This is slightly contradictory (removing decreases sum). 
        # Likely meant: Keep the highest 'a' until you have enough tokens.
        # Or: keep all, and if you have too many, remove lowest a.
        # Let's follow "Keep the highest 'a' until sum >= 1.005T".
        
        if not res: # Safety
            return [item[0] for item in C]
            
        return res

    # Final fail-safe
    return [item[0] for item in C]
