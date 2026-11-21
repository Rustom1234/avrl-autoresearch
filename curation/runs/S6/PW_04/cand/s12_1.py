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

    # 1. B = set of lowercased whitespace-word adjacent bigrams from all 200 target_sample docs
    B = set()
    for doc in target_sample:
        B.update(get_bigrams(doc))

    # 2. Split target docs into two halves by hash of the first 300 chars
    # Use a stable sort for repeatability with the seed
    target_sample.sort(key=lambda d: hash(d[:300] + str(seed)))
    h = len(target_sample) // 2
    half0 = target_sample[:h]
    half1 = target_sample[h:]

    # 3. Coverage calculations (not used directly in final formula but used for intermediate logic)
    # 4. a0 = dict of half-0 doc id -> fraction of bigrams in B
    def get_score(doc, bigram_set):
        b_in_doc = get_bigrams(doc)
        if not b_in_doc: return 0.0
        count = sum(1 for b in b_in_doc if b in bigram_set)
        return count / len(b_in_doc)

    a0_raw = [get_score(doc, B) for doc in half0]
    a1_raw = [get_score(doc, B) for doc in half1]

    # 6. a_LOO = per doc mean of (a0 scored with B1) + (a1 scored with B0) / 2
    B0 = set()
    for doc in half0: B0.update(get_bigrams(doc))
    B1 = set()
    for doc in half1: B1.update(get_bigrams(doc))

    a_loo_list = []
    for i in range(len(half0)):
        a_loo_list.append(get_score(half0[i], B1))
    for i in range(len(half1)):
        a_loo_list.append(get_score(half1[i], B0))

    # 7. mu and sd
    mu = sum(a_loo_list) / 200
    var = sum((x - mu)**2 for x in a_loo_list) / (200 - 2)
    sd = np.sqrt(var * 200 / (200 - 2))

    # 8. thr
    thr0 = sum(a0_raw) / len(half0)
    thr1 = sum(a1_raw) / len(half1)
    thr = max(0.2, 0.5 * thr0 + 0.5 * thr1 + 0.5 * sd)

    # 9. C0, C1, C
    def is_valid(doc, score):
        if len(doc) < 200: return False
        # Use zlib compression ratio as a proxy for duplication/entropy
        data = doc.encode()
        ratio = len(zlib.compress(data)) / len(data)
        return ratio <= 0.25 and score >= thr and score >= 0.1

    C0_list = []
    for i, doc in enumerate(half0):
        if is_valid(doc, a0_raw[i]):
            C0_list.append((i, doc))
            
    C1_list = []
    for i, doc in enumerate(half1):
        if is_valid(doc, a1_raw[i]):
            C1_list.append((h + i, doc))

    C_dict = {}
    for idx, doc in C0_list: C_dict[doc] = idx
    for idx, doc in C1_list:
        if doc not in C_dict: C_dict[doc] = idx

    # 10. Rebuild if len(C) < 50
    if len(C_dict) < 50:
        thr_low = max(0.1, 0.5 * thr0 + 0.5 * thr1 + 0.25 * sd)
        C_new = {}
        for i, doc in enumerate(half0):
            if is_valid(doc, a0_raw[i]): C_new[doc] = i
        for i, doc in enumerate(half1):
            if is_valid(doc, a1_raw[i]): C_new[doc] = h + i
        
        if len(C_new) < 50:
            all_scores = sorted([(a0_raw[i], i) for i in range(h)] + [(a1_raw[i], h+i) for i in range(h)], 
                                key=lambda x: x[0], reverse=True)
            for score, idx in all_scores:
                if docs[idx] not in C_new:
                    C_new[docs[idx]] = idx
                    if len(C_new) >= 20: break
            C_dict = C_new

    # 11. a = (mean(C0_a) + mean(C1_a) + 2*mu) / 3
    # We need the actual C0 and C1 sets as defined in step 9 to get these means
    # but if we used the rebuild logic, we use the C_new.
    # Let's stick to the logic of the current sets.
    
    # Re-extract C0 and C1 to ensure correct means for the specific formula
    c0_indices = [i for i, doc in enumerate(half0) if doc in C_dict and i < h]
    # The prompt implies we need the sets that formed C.
    # Let's calculate mean of a0 over C0 and a1 over C1.
    
    # Since C is the union of C0 and C1, let's identify them:
    # Let's re-evaluate C0 and C1 sets from the logic before C_dict merge.
    # (This handles the "mean of a0 over C0" correctly).
    
    # If rebuild wasn't triggered, C0 and C1 are the subsets of half0 and half1.
    # If rebuild WAS triggered, we use C_new.
    
    # To be safe and follow instructions:
    def get_means_from_final_c(final_c_dict):
        # Because the prompt doesn't specify how to handle C0/C1 if docs 
        # are shared or if rebuild happened, the most robust interpretation is 
        # the means of the scores of docs in the final sets.
        
        # Let's calculate means for the sets actually used to build C_dict
        # But the prompt says "mean of a0 over C0 plus mean of a1 over C1"
        # where C0 = {docs in half0 meeting criteria}
        # Let's re-calculate those specific sets:
        C0_final_vals = [a0_raw[i] for i, doc in enumerate(half0) if doc in C_dict and i < h]
        C1_final_vals = [a1_raw[i] for i, doc in enumerate(half1) if doc in C_dict and i >= h]
        
        m0 = sum(C0_final_vals) / len(C0_final_vals) if C0_final_vals else 0
        m1 = sum(C1_final_vals) / len(C1_final_vals) if C1_final_vals else 0
        return (m0 + m1 + 2 * mu) / 3

    a_final = get_means_from_final_c(C_dict)

    # Final selection
    # The prompt doesn't specify how to use a_final, but typically 
    # we use it to score docs in the pool or select the most relevant.
    # Given "select returns an ordered list of indices", and we need 
    # to fill T tokens, we score all docs in the pool.
    
    pool_scores = []
    for i, doc in enumerate(docs):
        # Score is basically how much it resembles target_sample
        score = get_score(doc, B)
        # We use a_final as a multiplier or threshold? 
        # Usually, these prompts imply using 'a' to rank the pool.
        pool_scores.append((score * a_final, i))
    
    # Sort by score descending
    pool_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Get top docs to fill T tokens
    # Since we need 9,830,400 tokens, and we want to avoid 
    # repeating too many times, but we can repeat.
    # Let's pick the best doc and repeat it if necessary, 
    # but the prompt says "repeats allowed" and the training is T tokens.
    # A standard approach is to take docs until T is reached.
    
    indices = []
    current_tokens = 0
    # To be efficient, we'll cycle the best docs.
    # Let's take the top 100 docs to ensure variety.
    top_indices = [idx for score, idx in pool_scores[:100]]
    
    # Calculate token counts (approximate using len / 4 as avg word is 4-5 chars)
    # Or just use a simple distribution.
    token_counts = [len(docs[i]) // 4 for i in top_indices]
    
    for idx in top_indices:
        count = token_counts[top_indices.index(idx)]
        # Add doc as many times as needed to approach T
        # But we must ensure we don't exceed T significantly.
        # The prompt says "repeats allowed".
        # Let's fill 90% of T with these 100 docs.
        needed = T - current_tokens
        if needed <= 0: break
        
        # Use this doc to fill as much as possible but keep it moving
        # To keep it simple: 
        if current_tokens + count <= T:
            indices.append(idx)
            current_tokens += count
        else:
            # Add partial
            indices.append(idx)
            current_tokens = T
            break

    # If we didn't reach T (unlikely with 2T pool), fill with the best
    if current_tokens < T:
        best_idx = pool_scores[0][1]
        while current_tokens < T:
            indices.append(best_idx)
            current_tokens += 1 # simple increment

    # The trainer reads docs in order and stops after T.
    # We must return indices such that the sum of tokens in docs[indices] >= T.
    # But the requirement is simply "returns an ordered list of indices".
    # The trainer will stop at T.
    
    # Wait, if the trainer stops at T, we should return enough indices
    # to cover at least T tokens.
    final_indices = []
    total_t = 0
    for idx in top_indices:
        count = len(docs[idx]) // 4
        final_indices.append(idx)
        total_t += count
        if total_t >= T:
            break
            
    # If we still haven't reached T, repeat the last one
    while total_t < T:
        final_indices.append(top_indices[-1])
        total_t += (len(docs[top_indices[-1]]) // 4)
        
    return final_indices
