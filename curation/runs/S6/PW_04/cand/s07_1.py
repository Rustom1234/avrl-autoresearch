import os
import random
import zlib
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_bigrams(text):
        words = text.lower().split()
        return set(zip(words, words[1:])) if len(words) > 1 else set()

    def get_a(doc, bigram_set):
        words = doc.lower().split()
        if not words: return 0.0
        limit = 3000
        count = 0
        match = 0
        for i in range(min(len(words) - 1, limit - 1)):
            bg = (words[i], words[i+1])
            count += 1
            if bg in bigram_set:
                match += 1
        return match / count if count > 0 else 0.0

    # 1. B = set of lowercased whitespace-word adjacent bigrams over the full 200 target_sample docs
    B = set()
    for doc in target_sample:
        B.update(get_bigrams(doc))

    # 2. Split the target docs into two halves by hash of the first 200 chars
    target_docs = list(target_sample)
    indexed_targets = []
    for i, doc in enumerate(target_docs):
        h = hash(doc[:200])
        indexed_targets.append((h, i))
    
    indexed_targets.sort(key=lambda x: x[0])
    half = len(target_docs) // 2
    train_indices = indexed_targets[:half]
    test_indices = indexed_targets[half:]
    
    train_docs = [target_docs[i] for h, i in train_indices]
    test_docs = [target_docs[i] for h, i in test_indices]

    # 3. a0 and a1 calculation
    B_test = set()
    for doc in test_docs:
        B_test.update(get_bigrams(doc))
        
    # Calculate a scores for all docs in the pool
    scores = []
    for doc in docs:
        scores.append(get_a(doc, B))

    a0 = np.mean([get_a(doc, B) for doc in train_docs]) if train_docs else 0.0
    a1 = np.mean([get_a(doc, B_test) for doc in test_docs]) if test_docs else 0.0

    # 5. thr calculation
    thr = max(0.33, a0 + 0.5 * (a0 - a1))

    # 6. Filter C
    def filter_docs(current_thr):
        indices = []
        seen_hashes = set()
        for i, doc in enumerate(docs):
            if len(doc) < 200: continue
            h = hash(doc)
            if h in seen_hashes: continue
            
            compressed = zlib.compress(doc.encode('utf-8'))
            ratio = len(compressed) / len(doc)
            
            if ratio <= 0.25 and scores[i] >= current_thr:
                indices.append(i)
                seen_hashes.add(h)
        return indices

    C_indices = filter_docs(thr)

    # If len(C) < 10, lower threshold
    if len(C_indices) < 10:
        new_thr = max(0.25, a0 + 0.2 * (a0 - a1))
        C_indices = filter_docs(new_thr)

    # 7. Recursive threshold adjustment
    while True:
        b = np.mean([scores[i] for i in C_indices]) if C_indices else 0
        if b >= a0 or not C_indices:
            break
        thr_new = max(0.25, a0 + 0.2 * (a0 - a1))
        C_indices = filter_docs(thr_new)

    # 8. T0 calculation
    T0 = sum(len(docs[i]) / 4 for i in C_indices)
    if T0 < T:
        T0 *= 1.1

    # 9. Check T0 vs T
    if T0 < 1.005 * T:
        res = list(C_indices)
        rng.shuffle(res)
        return res

    # 10. Descending order of a, remove bottom until T0 reached
    C_indices.sort(key=lambda i: scores[i], reverse=True)
    final_indices = []
    current_T0 = 0
    for i in C_indices:
        final_indices.append(i)
        current_T0 += len(docs[i]) / 4
        if current_T0 >= 1.005 * T:
            break
    
    if not final_indices: # Fallback for safety
        return []
        
    rng.shuffle(final_indices)
    return final_indices
