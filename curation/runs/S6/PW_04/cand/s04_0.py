import os
import random
import zlib
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-process target samples for bigram set
    def get_bigrams(text):
        words = text.lower().split()
        return set(zip(words[i], words[i+1]) for i in range(len(words)-1))

    Q = set()
    target_lengths = []
    for ts in target_sample:
        Q.update(get_bigrams(ts))
        # Calculate bigram fraction for target sample for b/sd
        words = ts.lower().split()
        if len(words) > 1:
            bigrams = list(zip(words[i], words[i+1]) for i in range(len(words)-1))
            count = sum(1 for b in bigrams if b in Q)
            target_lengths.append(count / len(bigrams))
        else:
            target_lengths.append(0.0)
        target_lengths.append(len(ts)) # Use length for T0 later if needed? No, prompt says sum len(t)/4.

    # Re-calculating target lengths based on prompt: T0 is sum len(t)/4.
    # Wait, prompt says T0 = sum len(t)/4 over top. Let's just get lengths.
    target_lens = [len(ts) for ts in target_sample]
    
    # Target stats for b and sd
    # The prompt says "mean of Q-fraction on the 200 target docs"
    # Re-calculating target_lengths to be just the fractions
    target_fractions = []
    for ts in target_sample:
        words = ts.lower().split()
        if len(words) > 1:
            count = 0
            for i in range(len(words)-1):
                if (words[i], words[i+1]) in Q:
                    count += 1
            target_fractions.append(count / (len(words)-1))
        else:
            target_fractions.append(0.0)
    
    b = np.mean(target_fractions)
    sd = np.std(target_fractions)
    
    # Process pool
    unique_docs = set()
    processed_pool = []
    
    for doc in docs:
        # Basic filters
        words = doc.lower().split()
        if len(words) < 200:
            continue
        
        # Dedupe and zlib test
        norm_text = " ".join(words)
        z_ratio = len(zlib.compress(norm_text.encode()).decode()) / len(norm_text) if len(norm_text) > 0 else 1.0
        if z_ratio > 0.25:
            continue
        
        if norm_text in unique_docs:
            continue
        unique_docs.add(norm_text)
        
        # Calculate a
        count = 0
        limit = 3000
        total_bigrams = 0
        for i in range(min(len(words)-1, limit)):
            bigram = (words[i], words[i+1])
            if bigram in Q:
                count += 1
            total_bigrams += 1
        
        a = count / total_bigrams if total_bigrams > 0 else 0
        processed_pool.append((a, doc, len(doc)))

    # Sort by a descending
    processed_pool.sort(key=lambda x: x[0], reverse=True)
    
    # Selection logic
    threshold_high = b + 0.5 * sd
    threshold_low = b + 0.2 * sd
    
    # Ensure threshold is not below 0.33
    if threshold_high < 0.33:
        threshold_high = 0.33
    if threshold_low < 0.33:
        threshold_low = 0.33

    top_indices = []
    for i, (a, doc, length) in enumerate(processed_pool):
        if a >= threshold_high:
            top_indices.append(i)
    
    if len(top_indices) < 10:
        for i, (a, doc, length) in enumerate(processed_pool):
            if i not in top_indices and a >= threshold_low:
                top_indices.append(i)
            if len(top_indices) == 10:
                break
        top_indices.sort() # keep original order for T0 calc or just use? 
        # Prompt says "take the next docs", implies maintaining order or just selection.
        # Let's sort them by 'a' descending to be safe for the "bottom" drop logic.
        top_indices = sorted(top_indices, key=lambda k: processed_pool[k][0], reverse=True)

    # T0 calculation
    t0 = sum(processed_pool[i][2] / 4.0 for i in top_indices)
    
    # Final trimming
    final_indices = []
    if t0 >= 1.005 * T:
        current_sum = 0
        # We need to drop from the bottom (lowest a)
        # top_indices is already sorted by a descending.
        # We keep from the top until we hit 1.005T, then we drop the rest.
        # Actually, "drop docs from the bottom until cumulative len/4 reaches 1.005T"
        # This means we keep the ones with highest 'a' first.
        # Since they are sorted descending by 'a', we just take from the start.
        for i in top_indices:
            current_sum += processed_pool[i][2] / 4.0
            final_indices.append(i)
            if current_sum >= 1.005 * T:
                break
    else:
        final_indices = top_indices

    # Map indices back to original docs
    # But select() needs indices into the original docs list.
    # Since we filtered, we need a way to map back.
    # Let's redefine processed_pool to store (a, original_index, length)
    # Re-doing logic slightly to keep track of original index.
    
    return None # Placeholder, see below for corrected logic

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
            
    b = np.mean(target_fractions)
    sd = np.std(target_fractions)
    
    processed_pool = []
    unique_docs = set()
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        if len(words) < 200: continue
        norm_text = " ".join(words)
        if norm_text in unique_docs: continue
        z_ratio = len(zlib.compress(norm_text.encode()).decode()) / len(norm_text) if len(norm_text) > 0 else 1.0
        if z_ratio > 0.25: continue
        unique_docs.add(norm_text)
        
        count = 0
        total_bigrams = 0
        for j in range(min(len(words)-1, 3000)):
            if (words[j], words[j+1]) in Q: count += 1
            total_bigrams += 1
        a = count / total_bigrams if total_bigrams > 0 else 0
        processed_pool.append((a, i, len(doc)))

    processed_pool.sort(key=lambda x: x[0], reverse=True)
    
    threshold_high = max(0.33, b + 0.5 * sd)
    threshold_low = max(0.33, b + 0.2 * sd)
    
    top_indices = []
    for i, (a, orig_idx, length) in enumerate(processed_pool):
        if a >= threshold_high:
            top_indices.append(i)
    
    if len(top_indices) < 10:
        for i, (a, orig_idx, length) in enumerate(processed_pool):
            if i not in top_indices and a >= threshold_low:
                top_indices.append(i)
                if len(top_indices) == 10: break
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
