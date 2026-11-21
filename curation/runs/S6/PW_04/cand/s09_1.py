import os
import random
import zlib
import numpy as np
from common import T

def get_doc_id(doc, length=300):
    content = doc[:length].encode('utf-8')
    return hash(content)

def get_dedupe_id(doc):
    content = doc.lower()[:500].encode('utf-8')
    return hash(content)

def get_bigrams(text):
    words = text.lower().split()
    bigrams = set()
    for i in range(len(words) - 1):
        bigrams.add((words[i], words[i+1]))
    return bigrams

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Build B
    B = set()
    for ts in target_sample:
        B.update(get_bigrams(ts))
        
    # 2. Split target docs
    half0_indices = []
    half1_indices = []
    for i, ts in enumerate(target_sample):
        if hash(ts[:200].encode('utf-8')) % 2 == 0:
            half0_indices.append(i)
        else:
            half1_indices.append(i)
            
    # 3. Compute a0, a1
    def get_scores(indices):
        scores = {}
        for idx in indices:
            ts = target_sample[idx]
            words = ts.lower().split()
            subset = words[:3000]
            count = 0
            total = 0
            for i in range(len(subset) - 1):
                bi = (subset[i], subset[i+1])
                if bi in B:
                    count += 1
                total += 1
            scores[idx] = count / total if total > 0 else 0.0
        return scores

    a0_dict = get_scores(half0_indices)
    a1_dict = get_scores(half1_indices)
    
    # 4. Calculate thr
    target_vals = []
    for i in range(len(target_sample)):
        if i in half0_indices:
            target_vals.append(a0_dict[i])
        else:
            target_vals.append(a1_dict[i])
    
    mean_a0 = np.mean(list(a0_dict.values())) if a0_dict else 0
    sd = np.std(target_vals) if target_vals else 0
    thr = max(0.33, mean_a0 + 0.5 * sd)

    # 5. C0, C1
    def filter_docs(docs, threshold):
        selected = {}
        for i, d in enumerate(docs):
            d_id = get_doc_id(d)
            dedupe_id = get_dedupe_id(d)
            words = d.lower().split()
            subset = words[:3000]
            count = 0
            total = 0
            for j in range(len(subset) - 1):
                bi = (subset[j], subset[j+1])
                if bi in B: count += 1
                total += 1
            score = count / total if total > 0 else 0.0
            
            z_ratio = len(zlib.compress(d.encode('utf-8'))) / len(d) if len(d) > 0 else 0
            
            if len(d) >= 200 and z_ratio <= 0.25 and score >= threshold:
                if dedupe_id not in selected:
                    selected[dedupe_id] = (i, d, d_id)
        return selected

    C = filter_docs(docs, thr)

    # 6. Rebuild logic
    if len(C) < 10:
        # Use new threshold logic
        new_thr = max(0.25, mean_a0 + 0.2 * sd)
        C_new = filter_docs(docs, new_thr)
        if len(C_new) < 10:
            # Take next by id
            sorted_indices = sorted(range(len(docs)), key=lambda i: get_doc_id(docs[i]))
            C_final = {}
            for idx in sorted_indices:
                d = docs[idx]
                ded_id = get_dedupe_id(d)
                if ded_id not in C_final:
                    C_final[ded_id] = (idx, d, get_doc_id(d))
                    if len(C_final) == 10:
                        break
            C = C_final
        else:
            C = C_new

    # Output ordered list of indices
    # The prompt implies we return an ordered list of indices into 'docs'
    # We'll preserve the order they were found in C (which is order of docs)
    # To be safe and deterministic, we'll sort by original index
    result_indices = [v[0] for v in C.values()]
    result_indices.sort()
    return result_indices
