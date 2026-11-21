import os
import random
import zlib
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Normalise and Exact-dedup
    seen_keys = set()
    deduped_indices = []
    norm_docs = []
    
    for i, doc in enumerate(docs):
        s = doc.strip()
        key = ' '.join(s.split()).lower()
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_indices.append(i)
            norm_docs.append(s)
            
    # 2. Wiki-like Gate Identification
    wiki_like_indices = []
    for i, s in enumerate(norm_docs):
        if len(s) < 200:
            continue
        
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        
        words = s.split()
        if len(words) > 0:
            count = sum(1 for t in words if t == ',' or t == '.')
            rate = count / len(words)
            if rate >= 0.03:
                is_wiki = True
                
        if is_wiki or has_special:
            wiki_like_indices.append(i) # index into norm_docs

    # 3. Damage Score Calculation
    def get_bigrams(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    S = set()
    for i in range(100):
        S.update(get_bigrams(target_sample[i]))
        
    h_vals_target = []
    for i in range(100, 200):
        bg = get_bigrams(target_sample[i])
        limit = 3000
        h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit)
        h_vals_target.append(h)
    
    m = np.median(h_vals_target) if h_vals_target else 0.5
    
    # Pre-calculate h for all wiki-like docs
    wiki_h_scores = []
    for i in wiki_like_indices:
        s = norm_docs[i]
        bg = get_bigrams(s)
        limit = 3000
        h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit)
        wiki_h_scores.append(h)

    # Statistics and Filtering
    def filter_docs(threshold_mult):
        threshold = threshold_mult * m
        current_survivors = []
        for idx_in_norm, h_score in zip(wiki_like_indices, wiki_h_scores):
            if h_score >= threshold:
                doc = docs[deduped_indices[idx_in_norm]]
                # Junk guards
                z_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc) if len(doc) > 0 else 0
                if z_ratio < 0.2: continue
                doc_lower = doc.lower()
                if any(p in doc_lower for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                    continue
                current_survivors.append((deduped_indices[idx_in_norm], h_score))
        return current_survivors

    # Collect thresholds for logging
    for mult in [0.3, 0.4, 0.6, 0.8]:
        res = filter_docs(mult)
        est_tokens = sum(len(docs[idx]) // 4 for idx, _ in res)
        # print(f"Threshold {mult*m:.4f}: Wiki={len(wiki_like_indices)}, Survivors={len(res)}, Tokens={est_tokens}/{T}")

    # Main selection (0.4 * m)
    survivors_data = filter_docs(0.4)
    
    # Overflow Guard
    total_est_tokens = sum(len(docs[idx]) // 4 for idx, _ in survivors_data)
    if total_est_tokens > T:
        survivors_data.sort(key=lambda x: x[1], reverse=True)
        final_selection = []
        current_tokens = 0
        for idx, score in survivors_data:
            if current_tokens + (len(docs[idx]) // 4) <= T:
                final_selection.append(idx)
                current_tokens += (len(docs[idx]) // 4)
            else:
                break
    else:
        final_selection = [idx for idx, score in survivors_data]

    # Fallbacks
    if len(final_selection) < 200:
        # Fallback 1: All wiki-like
        wiki_indices = [deduped_indices[i] for i in wiki_like_indices]
        if len(wiki_indices) < 200:
            final_selection = wiki_indices
        else:
            final_selection = wiki_indices
            
    if len(final_selection) < 200:
        # Fallback 2: All deduped
        final_selection = deduped_indices

    # Final sanity check for duplicates and shuffle
    final_selection = list(set(final_selection))
    rng.shuffle(final_selection)
    return [int(i) for i in final_selection]
