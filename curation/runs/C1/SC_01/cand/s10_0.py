import os
import random
import zlib
import numpy as np
import tiktoken
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def count_tokens(texts):
        try:
            return len(enc.encode_ordinary_batch(texts, num_threads=8))
        except:
            return sum(len(doc) // 4 for doc in texts)

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

    # 2. Wiki-like Gate
    wiki_like_indices = []
    for i, s in enumerate(norm_docs):
        if len(s) < 200: continue
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens = s.split()
        if tokens:
            rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens)
            if rate >= 0.03: is_wiki = True
        if is_wiki or has_special:
            wiki_like_indices.append(deduped_indices[i])

    # 3. H-Value logic
    def get_bigrams_raw(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    target_bigrams = set()
    for i in range(min(len(target_sample), 200)):
        target_bigrams.update(get_bigrams_raw(target_sample[i]))

    h_vals = []
    for i in range(100, min(200, len(target_sample))):
        bg = get_bigrams_raw(target_sample[i])
        limit = 3000
        h = sum(1 for b in bg[:limit] if b in target_bigrams) / min(len(bg), limit)
        h_vals.append(h)
    m = np.median(h_vals) if h_vals else 0.5

    # Survivor set A
    temp_survivors = []
    for i, s in enumerate(norm_docs):
        if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
            bg = get_bigrams_raw(s)
            limit = 3000
            h = sum(1 for b in bg[:limit] if b in target_bigrams) / min(len(bg), limit)
            if h >= 0.6 * m:
                temp_survivors.append(deduped_indices[i])

    # Junk Guards
    junk_phrases = ['click here to order', 'free shipping', 'best price guaranteed']
    final_survivors = []
    for idx in temp_survivors:
        doc = docs[idx]
        if len(doc) == 0: continue
        ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if ratio < 0.2: continue
        if any(p in doc.lower() for p in junk_phrases): continue
        final_survivors.append(idx)

    # Fallbacks
    if len(final_survivors) < 200:
        temp_survivors_relaxed = []
        for i, s in enumerate(norm_docs):
            if len(s) >= 200 and (s.startswith('=') or any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(s.split()) > 0 and (sum(1 for t in s.split() if t == ',' or t == '.') / len(s.split())) >= 0.03)):
                bg = get_bigrams_raw(s)
                h = sum(1 for b in bg[:3000] if b in target_bigrams) / min(len(bg), 3000)
                if h >= 0.4 * m:
                    temp_survivors_relaxed.append(deduped_indices[i])
        
        final_survivors = []
        for idx in temp_survivors_relaxed:
            doc = docs[idx]
            if len(doc) > 0 and (len(zlib.compress(doc.encode('utf-8'))) / len(doc)) >= 0.2:
                if not any(p in doc.lower() for p in junk_phrases):
                    final_survivors.append(idx)
        if len(final_survivors) < 200:
            final_survivors = wiki_like_indices
        if len(final_survivors) < 200:
            final_survivors = deduped_indices

    # 4. Token Counting and Padding P
    A_indices = final_survivors
    A_texts = [docs[i] for i in A_indices]
    tokA = count_tokens(A_texts)

    if tokA >= 0.8 * T:
        rng.shuffle(A_indices)
        print(f"|A|={len(A_indices)}, tokA/T={tokA/T:.4f}, |P|=0, tokens(P)/T=0, copies=1, min_g=0, med_g=0")
        return [int(i) for i in A_indices]

    # Build P
    p_candidates = []
    for i, doc in enumerate(norm_docs):
        # Must have been in deduped but failed wiki gate
        is_wiki_gate = False
        if len(doc) >= 200 and (doc.startswith('=') or any(x in doc for x in [' @-@ ', ' @,@ ', ' @.@ ']) or (len(doc.split()) > 0 and (sum(1 for t in doc.split() if t == ',' or t == '.') / len(doc.split())) >= 0.03)):
            is_wiki_gate = True
        
        if is_wiki_gate: continue
        
        # Hard filters
        if len(doc) < 300: continue
        ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if ratio < 0.3: continue
        junk = ['click here to order', 'free shipping', 'best price guaranteed', 'discount code', 'coupon', 'limited offer']
        if any(p in doc.lower() for p in junk): continue
        if '$' in doc: continue
        
        # ASCII %
        ascii_count = sum(1 for c in doc if c.isalpha() or c == ' ')
        if ascii_count / len(doc) < 0.8: continue
        
        # Unique lines
        lines = doc.splitlines()
        if len(lines) > 0 and len(set(lines)) / len(lines) < 0.7: continue
        
        # Scoring g
        bg_set = set()
        for i in range(min(len(target_sample), 200)):
            words = target_sample[i].lower().split()
            bg_set.update([(words[j], words[j+1]) for j in range(len(words)-1)])
        
        tokens_g = re.findall(r"\w+|[^\w\s]", doc.lower())
        bg_g = []
        for i in range(len(tokens_g)-1):
            bg_g.append((tokens_g[i], tokens_g[i+1]))
        
        score = 0
        count = 0
        for b in bg_g[:2000]:
            if b in bg_set: score += 1
            count += 1
        g = score / count if count > 0 else 0
        p_candidates.append((deduped_indices[i], g))

    p_candidates.sort(key=lambda x: x[1], reverse=True)
    
    P_indices = []
    P_texts = []
    current_tokP = 0
    budget = min(0.2 * T, T - tokA)
    
    for idx, g in p_candidates:
        t = count_tokens([docs[idx]])
        if current_tokP + t <= budget:
            P_indices.append(idx)
            P_texts.append(docs[idx])
            current_tokP += t
        else:
            break
            
    # Final Construction
    res_indices = []
    if P_indices:
        rng.shuffle(P_indices)
        res_indices.extend(P_indices)
    
    rng.shuffle(A_indices)
    res_indices.extend(A_indices)
    
    # Fill until 1.05*T
    current_total = count_tokens([docs[i] for i in res_indices])
    copies = 1
    while current_total < 1.05 * T:
        rng.shuffle(A_indices)
        res_indices.extend(A_indices)
        current_total = count_tokens([docs[i] for i in res_indices])
        copies += 1

    # Stats
    min_g = p_candidates[0][1] if p_candidates else 0
    med_g = np.median([x[1] for x in p_candidates]) if p_candidates else 0
    print(f"|A|={len(A_indices)}, tokA/T={tokA/T:.4f}, |P|={len(P_indices)}, tokens(P)/T={current_tokP/T:.4f}, copies={copies}, min_g={min_g:.4f}, med_g={med_g:.4f}")
    
    return [int(i) for i in res_indices]
