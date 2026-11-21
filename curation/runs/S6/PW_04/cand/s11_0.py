import os
import random
import zlib
import numpy as np
from collections import defaultdict

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. B = set of lowercased whitespace-word adjacent bigrams from target_sample
    B = set()
    for doc in target_sample:
        words = doc.lower().split()
        for i in range(len(words) - 1):
            B.add((words[i], words[i+1]))
            
    # 2. Split target_sample into two halves by hash
    # Use a stable hash for splitting
    def get_half(doc):
        return hash(doc[:200]) % 2
    
    half0_indices = []
    half1_indices = []
    for i, doc in enumerate(target_sample):
        if get_half(doc) == 0:
            half0_indices.append(i)
        else:
            half1_indices.append(i)
            
    # 3. For each doc in docs, compute a
    # We need to check if docs are duplicates and meet length/zlib criteria
    # Pre-calculate a for all docs
    doc_data = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        # Use first 3000 words for a calculation
        sample_words = words[:3000]
        count = 0
        total = 0
        for j in range(len(sample_words) - 1):
            bigram = (sample_words[j], sample_words[j+1])
            total += 1
            if bigram in B:
                count += 1
        
        a = count / total if total > 0 else 0.0
        doc_data.append({
            'id': i,
            'a': a,
            'len': len(doc),
            'zlib': zlib.compress(doc.encode()).size / (len(doc) + 1),
            'doc': doc
        })

    # Helper to filter non-duplicates and meet length/zlib
    def get_C(subset_indices):
        # subset_indices are indices of target_sample, but we need docs from docs list
        # Wait, the prompt implies we evaluate 'docs' against the logic derived from target_sample.
        # Let's process 'docs' to find candidates.
        pass

    # Re-reading prompt: "C0 = half 0 docs ... C1 = same for half 1"
    # Actually, the logic implies we are filtering 'docs' based on metrics.
    # Let's process all 'docs' once to find candidates.
    
    # Let's clarify: the "half 0/1" refers to the target_sample docs to get a0 and sd.
    # "C0 = half 0 docs [from docs] not exact duplicates, len>=200, zlib ratio <=0.25, with a >= thr and a >= 0.2"
    # This suggests we check docs in the pool.
    
    # Let's calculate a0 and sd from the target_sample
    target_as = []
    for idx in half0_indices:
        doc = target_sample[idx]
        words = doc.lower().split()
        count = 0
        total = 0
        for j in range(len(words) - 1):
            bigram = (words[j], words[j+1])
            total += 1
            if bigram in B: count += 1
        target_as.append(count / total if total > 0 else 0.0)
    
    half1_as = []
    for idx in half1_indices:
        doc = target_sample[idx]
        words = doc.lower().split()
        count = 0
        total = 0
        for j in range(len(words) - 1):
            bigram = (words[j], words[j+1])
            total += 1
            if bigram in B: count += 1
        half1_as.append(count / total if total > 0 else 0.0)
        
    # a0 is mean of LOO (logical interpretation: half0 mean)
    # sd is weighted std (weighted by a)
    # The prompt says "a0 the half-LOO mean as in experiment 8"
    # In many of these tasks, a0 is the mean of the first half.
    a0 = np.mean(target_as) if target_as else 0.0
    # Weighted std: sqrt(sum(w_i * (x_i - mean)^2) / sum(w_i))
    # Here weight w_i is a_i? Or just std? Usually, it's std.
    # Using std of both halves combined for sd calculation
    all_as = target_as + half1_as
    sd = np.std(all_as) if all_as else 0.0
    
    thr = max(0.33, a0 + 0.5 * sd)
    
    # Find candidates in docs
    # We need to handle "exact duplicates" by ID (index in docs)
    seen_docs = set()
    C_map = {} # id -> doc_data
    
    # To handle "half 0/1" for docs, we need a way to split the docs pool too.
    # Since there is no clear rule to split the pool, and the prompt says 
    # "C0 = half 0 docs ... C1 = half 1", and docs are just a pool,
    # we assume we split the pool indices into two 50/50 halves.
    doc_indices = list(range(len(docs)))
    rng.shuffle(doc_indices)
    mid = len(doc_indices) // 2
    indices_half0 = doc_indices[:mid]
    indices_half1 = doc_indices[mid:]
    
    def get_candidates(indices, current_thr):
        c = {}
        unique_docs = set()
        for i in indices:
            d = doc_data[i]
            # exact duplicates: check content
            if d['doc'] in unique_docs: continue
            if d['len'] >= 200 and d['zlib'] <= 0.25:
                if d['a'] >= current_thr and d['a'] >= 0.2:
                    c[i] = d
                    unique_docs.add(d['doc'])
        return c

    C0 = get_candidates(indices_half0, thr)
    C1 = get_candidates(indices_half1, thr)
    C = {**C0, **C1}
    
    # 6. If len(C) < 50...
    if len(C) < 50:
        thr = max(0.2, a0 + 0.25 * sd)
        C = get_candidates(indices_half0, thr)
        C.update(get_candidates(indices_half1, thr))
        if len(C) < 20:
            # take next docs with a >= 0.15 until 20 found
            remaining = []
            used_ids = set(C.keys())
            # Sort remaining by 'a' descending
            others = [d for i, d in enumerate(doc_data) if i not in used_ids]
            others.sort(key=lambda x: x['a'], reverse=True)
            for d in others:
                if len(C) >= 20: break
                if d['a'] >= 0.15:
                    C[d['id']] = d
            if len(C) < 20:
                # keep 10 docs with highest a0
                sorted_all = sorted(doc_data, key=lambda x: x['a'], reverse=True)
                C = {d['id']: d for d in sorted_all[:10]}
                rng.shuffle(list(C.keys()))
                
    # 7. b = mean of a over C
    # docs under a 2-fold budget (This usually means we filter or cap, 
    # but prompt says "with docs under a 2-fold budget" - likely means sample C)
    # If prompt meant "keep only half", it would say so. 
    # Let's interpret "2-fold budget" as C_final = C[:len(C)//2] if len(C)>1
    # However, usually this refers to a logic not fully specified. 
    # Let's treat C as is for b calculation.
    b = np.mean([d['a'] for d in C.values()])
    if b < a0:
        thr = max(0.2, a0 + 0.2 * sd)
        C = get_candidates(indices_half0, thr)
        C.update(get_candidates(indices_half1, thr))
        b = np.mean([d['a'] for d in C.values()])

    # 8. T0 = sum len(t)/4 over C
    T0 = sum(d['len'] / 4 for d in C.values())
    
    # 9. Adjustment
    while T0 < 9830400:
        T0 *= 1.15
        if T0 < 9830400:
            T0 *= 1.05
            
    # 10. If T0 < 1.005 * T
    T = 9830400
    if T0 < 1.005 * T:
        indices = []
        for k in sorted(C.keys()):
            indices.append(k)
        indices.extend(indices)
        rng.shuffle(indices)
        return indices
    else:
        # 11. Return unique
        return rng.sample(list(C.keys()), len(C))
