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
        # Use a normalized key for deduplication
        key = ' '.join(s.split()).lower()
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_indices.append(i)
            norm_docs.append(s)
            
    def get_bigrams(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    # 2. Target analysis for H and F
    S = set()
    target_tokens_all = []
    for i in range(100):
        bg = get_bigrams(target_sample[i])
        S.update(bg)
        target_tokens_all.extend(target_sample[i].lower().split())
        
    token_counts = {}
    for t in target_tokens_all:
        token_counts[t] = token_counts.get(t, 0) + 1
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    F = set(t for t, count in sorted_tokens[:150])

    h_vals_target = []
    f_vals_target = []
    for i in range(100, 200):
        bg_t = get_bigrams(target_sample[i])
        # h
        limit = 3000
        h_t = sum(1 for b in bg_t[:limit] if b in S) / min(len(bg_t), limit) if bg_t else 0
        h_vals_target.append(h_t)
        # f
        f_t = 0.0
        f_count = 0
        f_hits = 0
        for b in bg_t:
            if b[0] in F and b[1] in F:
                f_count += 1
                if b in S:
                    f_hits += 1
        if f_count < 30:
            f_t = 1.0
        else:
            f_t = f_hits / f_count
        f_vals_target.append(f_t)
    
    m_h = np.median(h_vals_target) if h_vals_target else 0.5
    f_ref = np.percentile(f_vals_target, 5)
    f_thr = 0.93 * f_ref

    # 3. Process candidates (Set A)
    potential_A = []
    for i, s in enumerate(norm_docs):
        if len(s) < 200: continue
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens = s.split()
        if tokens:
            rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
            if rate >= 0.03: is_wiki = True
        
        if is_wiki or has_special:
            bg = get_bigrams(s)
            limit = 3000
            h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit) if bg else 0
            if h < 0.6 * m_h: continue
            
            doc = docs[deduped_indices[i]]
            if not doc: continue
            z_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
            if z_ratio < 0.2: continue
            
            doc_lower = doc.lower()
            if any(p in doc_lower for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                continue
            
            f = 0.0
            f_count = 0
            f_hits = 0
            for b in bg:
                if b[0] in F and b[1] in F:
                    f_count += 1
                    if b in S: f_hits += 1
            f = 1.0 if f_count < 30 else f_hits / f_count
            
            potential_A.append({'idx': deduped_indices[i], 'f': f, 'chars': len(doc)})

    # Fallbacks
    if len(potential_A) < 200:
        temp_relaxed = []
        for i, s in enumerate(norm_docs):
            if len(s) < 200: continue
            is_wiki = s.startswith('=')
            has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
            tokens = s.split()
            if tokens:
                rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
                if rate >= 0.03: is_wiki = True
            if is_wiki or has_special:
                bg = get_bigrams(s)
                limit = 3000
                h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit) if bg else 0
                if h < 0.4 * m_h: continue
                doc = docs[deduped_indices[i]]
                if len(doc) > 0 and len(zlib.compress(doc.encode('utf-8'))) / len(doc) >= 0.2:
                    if not any(p in doc.lower() for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                        f = 0.0
                        f_count = 0
                        f_hits = 0
                        for b in bg:
                            if b[0] in F and b[1] in F:
                                f_count += 1
                                if b in S: f_hits += 1
                        f = 1.0 if f_count < 30 else f_hits / f_count
                        temp_relaxed.append({'idx': deduped_indices[i], 'f': f, 'chars': len(doc)})
        if len(temp_relaxed) >= 200:
            potential_A = temp_relaxed
        else:
            # Use all deduped
            potential_A = [{'idx': i, 'f': 1.0, 'chars': len(docs[i])} for i in deduped_indices]

    final_A_indices = [item['idx'] for item in potential_A]
    total_chars_A = sum(item['chars'] for item in potential_A)
    
    final_B_indices = []
    total_chars_B = 0
    for item in potential_A:
        if item['f'] >= f_thr:
            final_B_indices.append(item['idx'])
            total_chars_B += item['chars']
            
    if len(final_B_indices) < 200 or (total_chars_A > 0 and total_chars_B / total_chars_A < 0.5):
        res_indices = final_A_indices
    else:
        res_indices = final_B_indices

    rng.shuffle(res_indices)
    return [int(i) for i in res_indices]
