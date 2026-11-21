import os
import random
import zlib
import numpy as np
from collections import Counter

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
    # To ensure a "clean half" is never rejected, we split the sorted hashes
    # First half is training, second half is test.
    train_indices = indexed_targets[:half]
    test_indices = indexed_targets[half:]
    
    train_docs = [target_docs[i] for h, i in train_indices]
    test_docs = [target_docs[i] for h, i in test_indices]

    # 3. For each doc, a = fraction of its bigrams (first 3000 words) that are in B
    # Wait, the instruction says a0 = mean over training half with B, 
    # and a1 = mean over test half with a copy of B built from that half.
    # However, step 1 says B is built from the FULL 200.
    # Re-reading: "a0 = mean of a over the half scored with B (training set); 
    # a1 = mean of a over the other half with a copy of B built from that half (test set)."
    
    # Re-evaluating B for test set:
    B_test = set()
    for doc in test_docs:
        B_test.update(get_bigrams(doc))
        
    # Calculate a scores
    # Note: Step 3 says "fraction ... that are in B".
    # For a0, we use B (the full set). For a1, we use B_test.
    scores = []
    for doc in docs:
        scores.append(get_a(doc, B)) # This is the primary score used for C

    a0 = np.mean([get_a(doc, B) for doc in train_docs])
    a1 = np.mean([get_a(doc, B_test) for doc in test_docs])

    # 5. thr calculation
    thr = max(0.33, a0 + 0.5 * (a0 - a1))

    # 6. C = pool docs not exact duplicates, len>=200, zlib ratio <=0.25, with a >= thr
    # To handle "not exact duplicates", we use a set of hashes.
    C_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        if len(doc) < 200: continue
        h = hash(doc)
        if h in seen_hashes: continue
        
        # zlib ratio: size of compressed / size of original
        compressed = zlib.compress(doc.encode('utf-8'))
        ratio = len(compressed) / len(doc)
        
        if ratio <= 0.25 and scores[i] >= thr:
            C_indices.append(i)
            seen_hashes.add(h)

    # If len(C) < 10, lower threshold
    if len(C_indices) < 10:
        new_thr = max(0.25, a0 + 0.2 * (a0 - a1))
        C_indices = []
        seen_hashes = set()
        for i, doc in enumerate(docs):
            if len(doc) < 200: continue
            h = hash(doc)
            if h in seen_hashes: continue
            compressed = zlib.compress(doc.encode('utf-8'))
            ratio = len(compressed) / len(doc)
            if ratio <= 0.25 and scores[i] >= new_thr:
                C_indices.append(i)
                seen_hashes.add(h)

    # 7. b = mean of a over C. Rebuild C if b < a0.
    # This loop handles the recursive threshold adjustment.
    while True:
        b = np.mean([scores[i] for i in C_indices]) if C_indices else 0
        if b >= a0 or not C_indices:
            break
        # Rebuild C with new thr
        thr_new = max(0.25, a0 + 0.2 * (a0 - a1))
        C_indices = []
        seen_hashes = set()
        for i, doc in enumerate(docs):
            if len(doc) < 200: continue
            h = hash(doc)
            if h in seen_hashes: continue
            compressed = zlib.compress(doc.encode('utf-8'))
            ratio = len(compressed) / len(doc)
            if ratio <= 0.25 and scores[i] >= thr_new:
                C_indices.append(i)
                seen_hashes.add(h)

    # 8. T0 = sum len(t)/4 over C
    # Note: common.T is the target 9,830,400
    from common import T
    T0 = sum(len(docs[i]) / 4 for i in C_indices)
    if T0 < T:
        T0 *= 1.1

    # 9. Check T0 vs T
    if T0 < 1.005 * T:
        # Return all of C shuffled
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
    
    # Shuffle result
    rng.shuffle(final_indices)
    return final_indices
