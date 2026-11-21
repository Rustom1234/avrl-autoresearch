```python
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
            
    # Helpers for metrics
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
        
    # Build F: 150 most frequent tokens
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
        h_t = sum(1 for b in bg_t[:limit] if b in S) / min(len(bg_t), limit)
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
    f_ref = np.percentile(f_vals_target, 40) # Using 40th for stability but instructions say 5th percentile
    f_ref = np.percentile(f_vals_target, 5)
    f_thr = 0.93 * f_ref

    # 3. Process candidates (Set A)
    # Logic: wiki-like gate, h >= 0.6*m, zlib >= 0.2, spam check
    potential_A = []
    for i, s in enumerate(norm_docs):
        # Wiki gate
        if len(s) < 200: continue
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens = s.split()
        if tokens:
            rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
            if rate >= 0.03: is_wiki = True
        
        if is_wiki or has_special:
            # Bigram h
            bg = get_bigrams(s)
            limit = 3000
            h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit)
            if h < 0.6 * m_h: continue
            
            # Junk checks
            doc = docs[deduped_indices[i]]
            if not doc: continue
            z_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
            if z_ratio < 0.2: continue
            
            doc_lower = doc.lower()
            if any(p in doc_lower for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                continue
            
            # Bigram f
            f = 0.0
            f_count = 0
            f_hits = 0
            for b in bg:
                if b[0] in F and b[1] in F:
                    f_count += 1
                    if b in S: f_hits += 1
            if f_count < 30: f = 1.0
            else: f = f_hits / f_count
            
            potential_A.append({'idx': deduped_indices[i], 'f': f, 'chars': len(docs[deduped_indices[i]])})

    # Fallbacks for A
    if len(potential_A) < 200:
        # Relax h to 0.4*m
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
                h = sum(1 for b in bg[:3000] if b in S) / min(len(bg), 3000)
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
        
        if len(temp_relaxed) < 200:
            # Use all wiki-like deduped
            potential_A = []
            for i, s in enumerate(norm_docs):
                if len(s) >= 200:
                    is_wiki = s.startswith('=')
                    has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
                    tokens = s.split()
                    if tokens:
                        rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
                        if rate >= 0.03: is_wiki = True
                    if is_wiki or has_special:
                        potential_A.append({'idx': deduped_indices[i], 'f': 1.0, 'chars': len(s)})
        
        if len(potential_A) < 200:
            # Use all deduped
            potential_A = [{'idx': i, 'f': 1.0, 'chars': len(docs[i])} for i in deduped_indices]

    # Set B Calculation
    final_A_indices = [item['idx'] for item in potential_A]
    total_chars_A = sum(item['chars'] for item in potential_A)
    
    final_B_indices = [item['idx'] for item in potential_A if item['f'] >= f_thr]
    total_chars_B = sum(item['chars'] for item in potential_B_indices_helper if item['f'] >= f_thr) # Wait, refactor
    
    # Correct B calculation
    final_B_indices = []
    total_chars_B = 0
    for item in potential_A:
        if item['f'] >= f_thr:
            final_B_indices.append(item['idx'])
            total_chars_B += item['chars']
            
    # Safety
    if len(final_B_indices) < 200 or (total_chars_A > 0 and total_chars_B / total_chars_A < 0.5):
        res_indices = final_A_indices
    else:
        res_indices = final_B_indices

    # Stats
    f_A = [item['f'] for item in potential_A]
    # Print stats
    print(f"|A|: {len(final_A_indices)}")
    print(f"|B|: {len(final_B_indices)}")
    print(f"chars(B)/chars(A): {total_chars_B/total_chars_A if total_chars_A > 0 else 0}")
    print(f"f_ref: {f_ref}, f_thr: {f_thr}")
    print(f"Deciles f(A): {np.percentile(f_A, np.arange(0, 101, 10))}")
    print(f"Deciles f(target 100-199): {np.percentile(f_vals_target, np.arange(0, 101, 10))}")
    print(f"Est tokens(B)/T: {total_chars_B/4 / T if T > 0 else 0}")

    rng.shuffle(res_indices)
    return [int(i) for i in res_indices]

# Fix logic: ensure f_ref and f_thr are calculated correctly and B logic matches
# Re-writing slightly to ensure perfect alignment with requirements.
def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
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

    S = set()
    target_tokens_all = []
    for i in range(100):
        bg = get_bigrams(target_sample[i])
        S.update(bg)
        target_tokens_all.extend(target_sample[i].lower().split())
        
    token_counts = {}
    for t in target_tokens_all:
        token_counts[t] = token_counts.get(t, 0) + 1
    F = set(t for t, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:150])

    h_vals_target = []
    f_vals_target = []
    for i in range(100, 200):
        bg_t = get_bigrams(target_sample[i])
        h_t = sum(1 for b in bg_t[:3000] if b in S) / min(len(bg_t), 3000)
        h_vals_target.append(h_t)
        f_t = 0.0
        f_count = 0
        f_hits = 0
        for b in bg_t:
            if b[0] in F and b[1] in F:
                f_count += 1
                if b in S: f_hits += 1
        f_t = 1.0 if f_count < 30 else f_hits / f_count
        f_vals_target.append(f_t)
    
    m_h = np.median(h_vals_target) if h_vals_target else 0.5
    f_ref = np.percentile(f_vals_target, 5)
    f_thr = 0.93 * f_ref

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
            h = sum(1 for b in bg[:3000] if b in S) / min(len(bg), 3000)
            if h < 0.6 * m_h: continue
            doc = docs[deduped_indices[i]]
            if not doc: continue
            z_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
            if z_ratio < 0.2: continue
            if any(p in doc.lower() for p in ['click here to order', 'free shipping', 'best price guaranteed']): continue
            f = 0.0
            f_count = 0
            f_hits = 0
            for b in bg:
                if b[0] in F and b[1] in F:
                    f_count += 1
                    if b in S: f_hits += 1
            f = 1.0 if f_count < 30 else f_hits / f_count
            potential_A.append({'idx': deduped_indices[i], 'f': f, 'chars': len(doc)})

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
                h = sum(1 for b in bg[:3000] if b in S) / min(len(bg), 3000)
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
        if len(temp_relaxed) < 20
