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
        return set(zip(words[i], words[i+1]) for i in range(len(words)-1))

    Q = set()
    for ts in target_sample:
        Q.update(get_bigrams(ts))
    
    target_fractions = []
    for ts in target_sample:
        words = ts.lower().split()
        if len(words) > 1:
            count = sum(1 for i in range(len(words)-1) if (words[i], words[i+1]) in Q)
            target_fractions.append(count / (len(words)-1))
        else:
            target_fractions.append(0.0)
            
    b = np.mean(target_fractions) if target_fractions else 0.0
    sd = np.std(target_fractions) if target_fractions else 0.0
    
    processed_pool = []
    unique_docs = set()
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        if len(words) < 200: continue
        
        norm_text = " ".join(words)
        if norm_text in unique_docs: continue
        
        # Fix for UnicodeDecodeError: compress returns bytes, decode with ignore
        compressed = zlib.compress(norm_text.encode('utf-8', 'ignore'))
        try:
            z_ratio = len(compressed.decode('utf-8', 'ignore')) / len(norm_text)
        except:
            z_ratio = len(compressed) / len(norm_text)
            
        if z_ratio > 0.25: continue
        unique_docs.add(norm_text)
        
        count = 0
        total_bigrams = 0
        # Check bigrams
        for j in range(min(len(words)-1, 3000)):
            if (words[j], words[j+1]) in Q:
                count += 1
            total_bigrams += 1
        a = count / total_bigrams if total_bigrams > 0 else 0
        processed_pool.append((a, i, len(doc)))

    # Sort by a descending
    processed_pool.sort(key=lambda x: x[0], reverse=True)
    
    threshold_high = max(0.33, b + 0.5 * sd)
    threshold_low = max(0.33, b + 0.2 * sd)
    
    top_indices = []
    for idx, (a, orig_idx, length) in enumerate(processed_pool):
        if a >= threshold_high:
            top_indices.append(idx)
    
    if len(top_indices) < 10:
        for idx, (a, orig_idx, length) in enumerate(processed_pool):
            if idx not in top_indices and a >= threshold_low:
                top_indices.append(idx)
                if len(top_indices) == 10:
                    break
        top_indices = sorted(top_indices, key=lambda k: processed_pool[k][0], reverse=True)
    
    t0 = sum(processed_pool[i][2] / 4.0 for i in top_indices)
    
    final_top_indices = []
    if t0 >= 1.005 * T:
        curr_sum = 0
        for i in top_indices:
            curr_sum += processed_pool[i][2] / 4.0
            final_top_indices.append(i)
            if curr_sum >= 1.005 * T:
                break
    else:
        final_top_indices = top_indices
        
    res = [processed_pool[i][1] for i in final_top_indices]
    rng.shuffle(res)
    return res
