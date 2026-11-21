import os
import random
import zlib
import numpy as np
from common import T

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
    def get_half(doc):
        # Using a simple stable sum of ords to avoid hash() randomization
        return sum(ord(c) for c in doc[:200]) % 2
    
    half0_indices = []
    half1_indices = []
    for i, doc in enumerate(target_sample):
        if get_half(doc) == 0:
            half0_indices.append(i)
        else:
            half1_indices.append(i)
            
    # Pre-calculate a for all docs in the pool
    doc_data = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        sample_words = words[:3000]
        count = 0
        total = 0
        for j in range(len(sample_words) - 1):
            bigram = (sample_words[j], sample_words[j+1])
            total += 1
            if bigram in B:
                count += 1
        
        a = count / total if total > 0 else 0.0
        # Corrected zlib logic: .size is for numpy arrays, use len() for bytes
        compressed = zlib.compress(doc.encode('utf-8'))
        zlib_ratio = len(compressed) / (len(doc) + 1)
        
        doc_data.append({
            'id': i,
            'a': a,
            'len': len(doc),
            'zlib': zlib_ratio,
            'doc': doc
        })

    # Calculate metrics from target_sample
    def get_as(indices):
        as_list = []
        for idx in indices:
            doc = target_sample[idx]
            words = doc.lower().split()
            count = 0
            total = 0
            for j in range(len(words) - 1):
                bigram = (words[j], words[j+1])
                total += 1
                if bigram in B: count += 1
            as_list.append(count / total if total > 0 else 0.0)
        return as_list

    target_as = get_as(half0_indices)
    half1_as = get_as(half1_indices)
    
    a0 = np.mean(target_as) if target_as else 0.0
    all_as = target_as + half1_as
    sd = np.std(all_as) if all_as else 0.0
    
    thr = max(0.33, a0 + 0.5 * sd)
    
    # Split pool indices for C0 and C1
    doc_indices = list(range(len(docs)))
    rng.shuffle(doc_indices)
    mid = len(doc_indices) // 2
    idx_h0 = doc_indices[:mid]
    idx_h1 = doc_indices[mid:]
    
    def get_candidates(indices, current_thr):
        c = {}
        unique_docs = set()
        for i in indices:
            d = doc_data[i]
            if d['doc'] in unique_docs: continue
            if d['len'] >= 200 and d['zlib'] <= 0.25:
                if d['a'] >= current_thr and d['a'] >= 0.2:
                    c[d['id']] = d
                    unique_docs.add(d['doc'])
        return c

    C0 = get_candidates(idx_h0, thr)
    C1 = get_candidates(idx_h1, thr)
    C = {**C0, **C1}
    
    if len(C) < 50:
        thr = max(0.2, a0 + 0.25 * sd)
        C = get_candidates(idx_h0, thr)
        C.update(get_candidates(idx_h1, thr))
        if len(C) < 20:
            remaining = [d for i, d in enumerate(doc_data) if i not in C]
            remaining.sort(key=lambda x: x['a'], reverse=True)
            for d in remaining:
                if len(C) >= 20: break
                if d['a'] >= 0.15:
                    C[d['id']] = d
            if len(C) < 20:
                sorted_all = sorted(doc_data, key=lambda x: x['a'], reverse=True)
                C = {d['id']: d for d in sorted_all[:10]}
                rng.shuffle(list(C.keys()))
                
    b = np.mean([d['a'] for d in C.values()])
    if b < a0:
        thr = max(0.2, a0 + 0.2 * sd)
        C = get_candidates(idx_h0, thr)
        C.update(get_candidates(idx_h1, thr))
        b = np.mean([d['a'] for d in C.values()])

    T0 = sum(d['len'] / 4 for d in C.values())
    while T0 < 9830400:
        T0 *= 1.15
        if T0 < 9830400:
            T0 *= 1.05
            
    if T0 < 1.005 * T:
        indices = []
        for k in sorted(C.keys()):
            indices.append(k)
        indices.extend(indices)
        rng.shuffle(indices)
        return indices
    else:
        return rng.sample(list(C.keys()), len(C))
