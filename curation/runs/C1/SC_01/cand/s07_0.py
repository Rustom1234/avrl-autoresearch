import os
import random
import zlib
import numpy as np
import re
import tiktoken
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
            
    # Helpers
    def get_bigrams_simple(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    def get_bigrams_punct(text):
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    def count_tokens(texts):
        try:
            enc = tiktoken.get_encoding("gpt2")
            return len(enc.encode_ordinary_batch(texts, num_threads=8))
        except Exception:
            return sum(len(t) // 4 for t in texts)

    # 2. Filter A logic
    wiki_like_indices = []
    for i, s in enumerate(norm_docs):
        if len(s) < 200: continue
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens_s = s.split()
        if tokens_s:
            rate = sum(1 for t in tokens_s if t == ',' or t == '.') / len(tokens_s)
            if rate >= 0.03: is_wiki = True
        if is_wiki or has_special:
            wiki_like_indices.append(i)

    S = set()
    for i in range(100):
        S.update(get_bigrams_simple(target_sample[i]))
        
    h_vals = []
    for i in range(100, 200):
        bg = get_bigrams_simple(target_sample[i])
        limit = 3000
        h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit) if bg else 0
        h_vals.append(h)
    m = np.median(h_vals) if h_vals else 0.5
    
    temp_survivors = []
    for i, s in enumerate(norm_docs):
        if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
            bg = get_bigrams_simple(s)
            limit = 3000
            h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit) if bg else 0
            if h >= 0.6 * m:
                temp_survivors.append(i)

    junk_phrases = ['click here to order', 'free shipping', 'best price guaranteed', 'discount code', 'coupon', 'limited offer']
    final_survivors_A = []
    for idx_in_norm in temp_survivors:
        doc = docs[deduped_indices[idx_in_norm]]
        if len(doc) == 0: continue
        ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if ratio < 0.2: continue
        if any(p in doc.lower() for p in junk_phrases): continue
        final_survivors_A.append(deduped_indices[idx_in_norm])

    # Fallbacks for A
    if len(final_survivors_A) < 200:
        temp_survivors_relaxed = []
        for i, s in enumerate(norm_docs):
            if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
                bg = get_bigrams_simple(s)
                h = sum(1 for b in bg[:3000] if b in S) / min(len(bg), 3000) if bg else 0
                if h >= 0.4 * m:
                    temp_survivors_relaxed.append(i)
        final_survivors_A = []
        for idx_in_norm in temp_survivors_relaxed:
            doc = docs[deduped_indices[idx_in_norm]]
            if len(doc) > 0 and (len(zlib.compress(doc.encode('utf-8'))) / len(doc)) >= 0.2:
                if not any(p in doc.lower() for p in junk_phrases):
                    final_survivors_A.append(deduped_indices[idx_in_norm])
        if len(final_survivors_A) < 200:
            final_survivors_A = wiki_like_indices
        if len(final_survivors_A) < 200:
            final_survivors_A = deduped_indices

    texts_A = [docs[i] for i in final_survivors_A]
    tokens_A_count = count_tokens(texts_A)
    
    if tokens_A_count >= T:
        rng.shuffle(final_survivors_A)
        return [int(i) for i in final_survivors_A]

    # 3. Build P
    P_candidates = []
    for i, idx in enumerate(deduped_indices):
        # Must have FAILED wiki gate
        is_wiki_gate = False
        s = norm_docs[i]
        if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
            is_wiki_gate = True
        
        if not is_wiki_gate:
            doc = docs[idx]
            if len(doc) < 300: continue
            ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
            if ratio < 0.3: continue
            if any(p in doc.lower() for p in junk_phrases): continue
            lines = doc.splitlines()
            if len(lines) == 0 or (len(set(lines)) / len(lines)) < 0.7: continue
            alpha = sum(1 for c in doc if c.isalpha() or c == ' ')
            if alpha / len(doc) < 0.8: continue
            P_candidates.append((idx, doc))

    # 4. Score P
    S2 = set()
    for ts in target_sample:
        S2.update(get_bigrams_punct(ts))
    
    scored_P = []
    for idx, doc in P_candidates:
        bg = get_bigrams_punct(doc)
        g = sum(1 for b in bg[:2000] if b in S2) / min(len(bg), 2000) if bg else 0
        scored_P.append((g, idx))
    
    scored_P.sort(key=lambda x: x[0], reverse=True)
    
    # Select P
    P_indices = []
    current_tokens = tokens_A_count
    for g, idx in scored_P:
        p_doc = docs[idx]
        p_tokens = len(p_doc) // 4 # Estimate for speed
        if current_tokens + p_tokens <= 1.02 * T:
            P_indices.append(idx)
            current_tokens += p_tokens
        else:
            break
    
    # 5. Order
    rng.shuffle(final_survivors_A)
    mid = len(final_survivors_A) // 2
    A1 = final_survivors_A[:mid]
    A2 = final_survivors_A[mid:]
    
    rng.shuffle(P_indices)
    result = [int(i) for i in P_indices] + [int(i) for i in A1] + [int(i) for i in A2]
    
    # 6. Print stats
    p_gs = [x[0] for x in scored_P]
    if p_gs:
        print(f"|A|: {len(final_survivors_A)}, tokens(A)/T: {tokens_A_count/T:.4f}, "
              f"|P|: {len(P_indices)}, tokens(P)/T: {current_tokens/T:.4f}, "
              f"min_g: {min(p_gs):.4f}, median_g: {np.median(p_gs):.4f}")
    else:
        print(f"|A|: {len(final_survivors_A)}, tokens(A)/T: {tokens_A_count/T:.4f}, "
              f"|P|: 0, tokens(P)/T: 0.0000, min_g: 0.0000, median_g: 0.0000")
              
    return result
