import os
import random
import zlib
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

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

    # Build Set S from first 100 target samples
    S = set()
    for i in range(min(100, len(target_sample))):
        S.update(get_bigrams(target_sample[i]))

    # Helpers for filtering
    def is_wiki_like(s):
        if len(s) < 200: return False
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens = s.split()
        if len(tokens) > 0:
            rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
            if rate >= 0.03: is_wiki = True
        return is_wiki or has_special

    def get_h(s):
        bg = get_bigrams(s)
        limit = 3000
        count = sum(1 for b in bg[:limit] if b in S)
        denom = min(len(bg), limit)
        return count / denom if denom > 0 else 0

    # Map deduped docs to metadata
    doc_metadata = []
    for i, s in enumerate(norm_docs):
        h = get_h(s)
        doc_metadata.append({
            'idx': deduped_indices[i],
            'norm': s,
            'h': h,
            'wiki': is_wiki_like(s)
        })

    # Calculate Median H for survivability
    h_vals_wiki = [m['h'] for m in doc_metadata if m['wiki']]
    m_h = np.median(h_vals_wiki) if h_vals_wiki else 0.5

    # Filter A
    A_indices = []
    for i, m in enumerate(doc_metadata):
        if m['wiki'] and m['h'] >= 0.6 * m_h:
            doc = docs[m['idx']]
            if len(doc) == 0: continue
            ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
            if ratio < 0.2: continue
            doc_lower = doc.lower()
            if any(p in doc_lower for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                continue
            A_indices.append(i)

    # Calculate Tokens for A
    texts_A = [doc_metadata[i]['norm'] for i in A_indices]
    # Fixed: Use encode() instead of encode_batch() to avoid the set subtraction error
    tokA = sum(len(enc.encode(t, allowed_special=False)) for t in texts_A)
    
    # Padding P
    P_indices = []
    if tokA < T:
        is_in_A = set(A_indices)
        candidates = []
        for i, m in enumerate(doc_metadata):
            if i in is_in_A: continue
            s = m['norm']
            if len(s) < 200: continue
            
            doc = docs[m['idx']]
            if len(doc) == 0: continue
            ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
            if ratio < 0.2: continue
            
            doc_lower = doc.lower()
            if any(p in doc_lower for p in ['click here to order', 'free shipping', 'best price guaranteed']):
                continue
                
            lines = s.splitlines()
            if len(lines) > 1:
                counts = {}
                for l in lines:
                    l_clean = l.strip()
                    counts[l_clean] = counts.get(l_clean, 0) + 1
                repeats = sum(1 for count in counts.values() if count > 1)
                if repeats / len(lines) > 0.3: continue
            
            candidates.append((m['h'], i))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        current_tok = tokA
        for h_val, idx in candidates:
            s = doc_metadata[idx]['norm']
            t_count = len(enc.encode(s, allowed_special=False))
            if current_tok + t_count <= 1.02 * T:
                P_indices.append(idx)
                current_tok += t_count
            else:
                break
    
    # Final Output Construction
    rng.shuffle(A_indices)
    mid = len(A_indices) // 2
    A1 = A_indices[:mid]
    A2 = A_indices[mid:]
    
    combined = P_indices + A1
    rng.shuffle(combined)
    
    final_indices = [doc_metadata[i]['idx'] for i in combined] + \
                     [doc_metadata[i]['idx'] for i in A2]
    
    return [int(i) for i in final_indices]
