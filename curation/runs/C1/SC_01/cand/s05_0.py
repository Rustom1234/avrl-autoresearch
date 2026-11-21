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
            
    def get_bigrams(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    def get_h1(text, S_ref):
        bg = get_bigrams(text)
        limit = 3000
        if not bg: return 0.0
        count = sum(1 for b in bg[:limit] if b in S_ref)
        return count / min(len(bg), limit)

    # Initial S for Pass 1
    S1 = set()
    for i in range(100):
        S1.update(get_bigrams(target_sample[i]))
        
    h1_vals = []
    # Pass 1 logic
    temp_pass1 = []
    for i, s in enumerate(norm_docs):
        # Loose wiki gate
        if len(s) < 200: continue
        is_wiki = s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens = s.split()
        if len(tokens) > 0:
            rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
            if rate >= 0.03: is_wiki = True
        
        if is_wiki:
            h1 = get_h1(s, S1)
            temp_pass1.append((deduped_indices[i], s, h1))

    # Calculate m for Pass 1
    h1_vals = [x[2] for x in temp_pass1]
    m1 = np.median(h1_vals) if h1_vals else 0.5
    
    # Filter h1 >= 0.6 * m1
    pass1_candidates = [x for x in temp_pass1 if x[2] >= 0.6 * m1]
    
    # Junk guards on pass1_candidates
    junk_phrases = ['click here to order', 'free shipping', 'best price guaranteed']
    A_indices = []
    for idx, s, h1 in pass1_candidates:
        doc = docs[idx]
        if not doc: continue
        if len(zlib.compress(doc.encode('utf-8'))) / len(doc) < 0.2: continue
        if any(p in doc.lower() for p in junk_phrases): continue
        A_indices.append(idx)

    # 2. Bootstrap S2
    # Use docs from A that meet h1 criteria
    med_h1_A = np.median(h1_vals) if h1_vals else 0.5 # Actually should be median of A's h1
    # Re-calculating median of h1 specifically for docs that passed A filter
    h1_A = [x[2] for x in pass1_candidates if x[0] in A_indices]
    m_h1_A = np.median(h1_A) if h1_A else 0.5
    
    S2 = set()
    for i in range(100):
        S2.update(get_bigrams(target_sample[i]))
    for idx, s, h1 in pass1_candidates:
        if h1 >= m_h1_A:
            S2.update(get_bigrams(s))
            
    # 3. Rescore
    h2_scores = {}
    # Docs that contributed to S2 automatically
    S2_contributors = set()
    for idx, s, h1 in pass1_candidates:
        if h1 >= m_h1_A:
            S2_contributors.add(idx)
            
    # 4. Calibrate threshold
    clean_h2 = []
    damaged_h2 = []
    for i in range(100, 200):
        doc_s = target_sample[i]
        h2_c = get_h1(doc_s, S2)
        clean_h2.append(h2_c)
        
        # Damage: shuffle tokens in blocks of 10
        words = doc_s.lower().split()
        damaged_words = []
        for j in range(0, len(words), 10):
            block = words[j:j+10]
            damaged_words.extend(rng.sample(block, len(block)))
        damaged_doc = " ".join(damaged_words)
        damaged_h2.append(get_h1(damaged_doc, S2))
    
    c5 = np.percentile(clean_h2, 5)
    d95 = np.percentile(damaged_h2, 95)
    if d95 < c5:
        thr = 0.5 * (c5 + d95)
    else:
        thr = 0.8 * c5
        
    # 5. Survivors B
    B_indices = []
    for idx in A_indices:
        if idx in S2_contributors:
            B_indices.append(idx)
        else:
            s = norm_docs[deduped_indices.index(idx)]
            if get_h1(s, S2) >= thr:
                B_indices.append(idx)
                
    # 6. Safety
    tokens_A = sum(len(docs[i]) // 4 for i in A_indices)
    tokens_B = sum(len(docs[i]) // 4 for i in B_indices)
    
    final_indices = []
    if len(B_indices) < 200 or (len(A_indices) > 0 and tokens_B < 0.6 * tokens_A):
        final_indices = A_indices
    else:
        final_indices = B_indices

    # Print stats
    pool_h2 = [get_h1(norm_docs[idx], S2) for idx in A_indices]
    print(f"|A|: {len(A_indices)}, |B|: {len(B_indices)}, thr: {thr}")
    print(f"clean q5: {c5}, damaged q95: {d95}, pool h2 mean: {np.mean(pool_h2) if pool_h2 else 0}")

    rng.shuffle(final_indices)
    return [int(i) for i in final_indices]
