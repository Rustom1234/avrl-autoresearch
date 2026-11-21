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
            
    # 2. Loose format gate
    wiki_like_indices = []
    for i, s in enumerate(norm_docs):
        if len(s) < 200:
            continue
        
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        
        # Rate of ' , ' plus ' . '
        tokens = s.split()
        if len(tokens) > 0:
            count = sum(1 for t in tokens if t == ',' or t == '.')
            rate = count / len(tokens)
            if rate >= 0.03:
                is_wiki = True
                
        if is_wiki or has_special:
            wiki_like_indices.append(deduped_indices[i])

    # 3. Damage filter
    def get_bigrams(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    S = set()
    for i in range(100):
        S.update(get_bigrams(target_sample[i]))
        
    h_vals = []
    for i in range(100, 200):
        bg = get_bigrams(target_sample[i])
        limit = 3000
        h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit)
        h_vals.append(h)
    
    m = np.median(h_vals) if h_vals else 0.5
    
    survivors = []
    for idx in wiki_like_indices:
        doc = norm_docs[deduped_indices.index(idx)]
        # Need to map back to the correct normalized string
        # The loop above already processed norm_docs
        pass
    
    # Re-evaluating logic for efficiency and correctness:
    temp_survivors = []
    for i, s in enumerate(norm_docs):
        # Re-check wiki-like criteria for all deduped
        if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
            bg = get_bigrams(s)
            limit = 3000
            h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit)
            if h >= 0.6 * m:
                temp_survivors.append(deduped_indices[i])

    # 4. Junk guards
    final_survivors = []
    junk_phrases = ['click here to order', 'free shipping', 'best price guaranteed']
    for idx in temp_survivors:
        doc = docs[idx]
        if len(doc) == 0: continue
        
        # zlib check
        ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if ratio < 0.2: continue
        
        # text check
        doc_lower = doc.lower()
        if any(phrase in doc_lower for phrase in junk_phrases):
            continue
            
        final_survivors.append(idx)

    # 5. Fallbacks
    if len(final_survivors) < 200:
        # Relax h
        temp_survivors_relaxed = []
        for i, s in enumerate(norm_docs):
            if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
                bg = get_bigrams(s)
                h = sum(1 for b in bg[:3000] if b in S) / min(len(bg), 3000)
                if h >= 0.4 * m:
                    temp_survivors_relaxed.append(deduped_indices[i])
        
        # Junk guards on relaxed
        final_survivors = []
        for idx in temp_survivors_relaxed:
            doc = docs[idx]
            if len(doc) > 0 and zlib.compress(doc.encode('utf-8')) / len(doc) >= 0.2:
                if not any(p in doc.lower() for p in junk_phrases):
                    final_survivors.append(idx)

        if len(final_survivors) < 200:
            # Use all wiki-like deduped
            final_survivors = wiki_like_indices
            
        if len(final_survivors) < 200:
            # Use all deduped
            final_survivors = deduped_indices

    # 6. Output
    rng.shuffle(final_survivors)
    return [int(i) for i in final_survivors]
