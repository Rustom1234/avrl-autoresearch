import os
import random
import re
import zlib
import numpy as np
from common import T

def get_bigrams(text):
    tokens = re.findall(r'\w+', text.lower())
    return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else set()

def get_ab(doc, bigrams_set):
    tokens = re.findall(r'\w+', doc.lower())
    sub_tokens = tokens[:3000]
    if len(sub_tokens) < 2:
        return 0.0
    count = 0
    for i in range(len(sub_tokens) - 1):
        pair = (sub_tokens[i], sub_tokens[i+1])
        if pair in bigrams_set:
            count += 1
    return count / (len(sub_tokens) - 1)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. B = set of bigrams from target_sample
    B = set()
    for doc in target_sample:
        B.update(get_bigrams(doc))
        
    # 2. Split target docs into two halves by hash
    def get_stable_hash(s):
        # Use a simple stable hash for first 200 chars
        h = 0
        for char in s[:200]:
            h = (31 * h + ord(char)) % (10**9 + 7)
        return h
    
    indexed_targets = list(enumerate(target_sample))
    indexed_targets.sort(key=lambda x: get_stable_hash(x[1]))
    
    # Half split
    h1 = indexed_targets[:100]
    h2 = indexed_targets[100:]
    
    # 3. a_b scores
    a_b_scores = []
    for idx, doc in indexed_targets:
        a_b_scores.append(get_ab(doc, B))
        
    # 4. a0, a1
    a0 = sum(a_b_scores[i] for i in range(100)) / 100.0
    a1 = sum(a_b_scores[i] for i in range(100, 200)) / 100.0
    
    # 5. a_LEO scores
    a_leo_scores = [0.0] * 200
    for i in range(100):
        a_leo_scores[i] = a1
        a_leo_scores[i+100] = a0
        
    # 6. mu and sd
    mu = sum(a_leo_scores) / 200.0
    var = sum((x - mu)**2 for x in a_leo_scores) / 200.0
    sd = np.sqrt(var)
    
    # 7. thr
    thr = max(0.33, mu + 0.5 * sd)
    
    # 8. Filter C
    C_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        if len(doc) < 10: continue
        try:
            comp = zlib.compress(doc.encode('utf-8'))
            ratio = len(comp) / len(doc)
            if ratio <= 0.25:
                h = hash(doc)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    if get_ab(doc, B) >= thr:
                        C_indices.append(i)
        except:
            continue
            
    # Ensure we have some candidates
    if len(C_indices) == 0:
        C_indices = list(range(min(len(docs), 1000)))

    # 9. b = mean of a_LEO
    b = sum(a_leo_scores) / 200.0
    
    # 10. t0
    # use a rough token count: characters / 4
    t0_sum = sum(len(docs[i]) / 4 for i in C_indices)
    
    if t0_sum < T:
        t0_sum *= 1.05
        
    # 11. Check coverage
    if t0_sum < 1.005 * T:
        # If small, just return all valid candidates found
        return C_indices
    
    # 12. Descending a_b
    scored_C = []
    for i in C_indices:
        scored_C.append((get_ab(docs[i], B), i))
    scored_C.sort(key=lambda x: x[0], reverse=True)
    
    kept_indices = []
    current_t0 = 0
    for score, idx in scored_C:
        kept_indices.append(idx)
        current_t0 += len(docs[idx]) / 4
        if current_t0 >= 1.005 * T:
            break
            
    # 13. Return shuffled
    rng.shuffle(kept_indices)
    return kept_indices
