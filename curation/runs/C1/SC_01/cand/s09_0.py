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
            
    # Function to calculate h
    def get_h(text, S):
        words = text.lower().split()
        bg = [(words[i], words[i+1]) for i in range(len(words)-1)]
        limit = 3000
        if not bg: return 0.0
        h = sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit)
        return h

    # Function to calculate c
    def get_c(text):
        toks = text.split()
        ok_count = 0
        total_positions = 0
        for i in range(len(toks) - 1):
            if toks[i] == '.':
                total_positions += 1
                nxt = toks[i+1]
                if nxt and (nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in ['=', '"', "'", '(', '[']):
                    ok_count += 1
        if total_positions < 5:
            return 1.0
        return ok_count / total_positions

    # Target Analysis
    S_target = set()
    target_h_vals = []
    target_c_vals = []
    for i in range(200):
        t_doc = target_sample[i]
        t_words = t_doc.lower().split()
        for j in range(len(t_words)-1):
            S_target.add((t_words[j], t_words[j+1]))
        # We only need c_ref for target_sample
        target_c_vals.append(get_c(t_doc))
    
    # Re-calculate S_target to be more robust (using first 100 as before)
    # Actually, the prompt says "Calibrate on all 200 target_sample docs"
    # Let's stick to the logic: h is based on S of first 100, c_ref on all 200
    S_h = set()
    for i in range(100):
        t_words = target_sample[i].lower().split()
        for j in range(len(t_words)-1):
            S_h.add((t_words[j], t_words[j+1]))
            
    c_ref = np.percentile(target_c_vals, 40) # 5th percentile is index 0 for 200 items? 
    # Wait, 5th percentile of 200 items is roughly index 10.
    c_ref = np.percentile(target_c_vals, 5)
    c_thr = 0.85 * c_ref
    
    # Processing deduped docs
    wiki_like_indices = []
    # We'll store (index, h, c)
    candidates = []
    
    for i, s in enumerate(norm_docs):
        # Loose wiki gate
        if len(s) < 200: continue
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        words = s.split()
        if len(words) > 0:
            rate = sum(1 for t in words if t == ',' or t == '.') / len(words)
            if rate >= 0.03: is_wiki = True
        
        if is_wiki or has_special:
            h = get_h(s, S_h)
            c = get_c(s)
            candidates.append((deduped_indices[i], h, c))

    # Determine Median h
    h_vals = [item[1] for item in candidates]
    m_h = np.median(h_vals) if h_vals else 0.5
    
    # Preliminary Set A: h >= 0.6*m
    A_indices = []
    for idx, h, c in candidates:
        if h >= 0.6 * m_h:
            A_indices.append(idx)
            
    # Apply zlib and junk filters to A to get A_filtered
    A_filtered = []
    junk_phrases = ['click here to order', 'free shipping', 'best price guaranteed']
    for idx in A_indices:
        doc = docs[idx]
        if len(doc) == 0: continue
        ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if ratio < 0.2: continue
        doc_lower = doc.lower()
        if any(p in doc_lower for p in junk_phrases): continue
        A_filtered.append(idx)

    # Now apply c >= c_thr to get B
    B_indices = []
    # We need to match back to c values
    # Let's refine the logic: A is the set of docs passing h-gate, zlib, and junk
    # B is the subset of A passing c-gate
    
    # Let's re-run the loop cleanly
    A_final = []
    B_final = []
    
    # Re-evaluate candidates to find valid A
    # Note: The prompt says A is the "current survivor set" (h >= 0.6*m + zlib + junk)
    # and B is "docs of the current survivor set A with c >= c_thr"
    
    for idx, h, c in candidates:
        if h >= 0.6 * m_h:
            doc = docs[idx]
            if len(doc) > 0:
                ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
                if ratio >= 0.2 and not any(p in doc.lower() for p in junk_phrases):
                    A_final.append(idx)
                    if c >= c_thr:
                        B_final.append(idx)

    # Safety check
    total_chars_A = sum(len(docs[i]) for i in A_final)
    total_chars_B = sum(len(docs[i]) for i in B_final)
    
    result = []
    if len(B_final) >= 200 and total_chars_B >= 0.5 * total_chars_A:
        result = B_final
    else:
        result = A_final
        
    # Fallbacks if result is too small
    if len(result) < 200:
        # Fallback 1: Relax h to 0.4 * m
        relaxed_A = []
        for idx, h, c in candidates:
            if h >= 0.4 * m_h:
                doc = docs[idx]
                if len(doc) > 0:
                    ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
                    if ratio >= 0.2 and not any(p in doc.lower() for p in junk_phrases):
                        relaxed_A.append(idx)
        if len(relaxed_A) >= 200:
            result = relaxed_A
        else:
            # Fallback 2: All wiki-like deduped
            wiki_only = []
            for idx, h, c in candidates:
                if h >= 0.6 * m_h: # Re-using wiki-gate criteria
                    wiki_only.append(idx)
            if len(wiki_only) >= 200:
                result = wiki_only
            else:
                result = deduped_indices

    # Final dedupe of result to ensure uniqueness
    unique_result = list(dict.fromkeys(result))
    
    # Printing requirements
    c_vals_A = [get_c(docs[i]) for i in A_final]
    c_vals_B = [get_c(docs[i]) for i in B_final]
    
    # Calculate counts for printing
    c_min = 0.5
    count_low = sum(1 for c in c_vals_A if c < c_min)
    count_mid = sum(1 for c in c_vals_A if c_min <= c < c_thr)
    count_high = sum(1 for c in c_vals_A if c >= c_thr)
    
    print(f"|A|: {len(A_final)}")
    print(f"|B|: {len(B_final)}")
    if total_chars_A > 0:
        print(f"chars(B)/chars(A): {total_chars_B / total_chars_A}")
    print(f"c_ref: {c_ref}")
    print(f"c_thr: {c_thr}")
    if c_vals_A:
        print(f"Deciles of c over A: {np.percentile(c_vals_A, np.arange(0, 100, 10))}")
    print(f"Counts (A): <0.5: {count_low}, [0.5, c_thr): {count_mid}, >=c_thr: {count_high}")

    rng.shuffle(unique_result)
    return [int(i) for i in unique_result]
