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
            
    # Helper to get bigrams
    def get_bigrams(text):
        words = text.lower().split()
        return [(words[i], words[i+1]) for i in range(len(words)-1)]

    # Calibration Data
    S = set()
    for i in range(100):
        S.update(get_bigrams(target_sample[i]))
    
    # Calibration for h
    h_vals_cal = []
    for i in range(100, 200):
        bg = get_bigrams(target_sample[i])
        limit = 3000
        h_vals_cal.append(sum(1 for b in bg[:limit] if b in S) / min(len(bg), limit) if len(bg) > 0 else 0)
    m = np.median(h_vals_cal) if h_vals_cal else 0.5

    # Calibration for c (Sentence-start capitalisation)
    c_vals_cal = []
    for i in range(200):
        toks = target_sample[i].split()
        ok = 0
        count = 0
        for j in range(len(toks)-1):
            if toks[j] == '.':
                count += 1
                next_t = toks[j+1]
                if next_t and (next_t[0].isupper() or next_t[0].isdigit() or next_t[0] in ['=', '"', "'", '(', '[']):
                    ok += 1
        c_vals_cal.append(ok / count if count > 0 else 1.0)
    c_vals_cal.sort()
    c_thr = 0.85 * (c_vals_cal[int(len(c_vals_cal) * 0.05)]) if c_vals_cal else 0
    
    # Calibration for f (Function-word bigram)
    F = set()
    all_words = []
    for i in range(100):
        all_words.extend(target_sample[i].lower().split())
    all_words.sort(key=lambda x: (x.isalpha(), x), reverse=True)
    # Simple approximation for 150 most frequent whitespace tokens
    counts = {}
    for w in all_words:
        counts[w] = counts.get(w, 0) + 1
    sorted_f_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    F = set(w for w, count in sorted_f_words[:150])
    
    f_vals_cal = []
    for i in range(100, 200):
        bg = get_bigrams(target_sample[i])
        f_words_bg = [b for b in bg if b[0] in F and b[1] in F]
        f_val = sum(1 for b in f_words_bg if b in S) / len(f_words_bg)
        f_vals_cal.append(f_val if len(f_words_bg) >= 30 else 1.0)
    f_vals_cal.sort()
    f_thr = 0.93 * (f_vals_cal[int(len(f_vals_cal) * 0.05)]) if f_vals_cal else 0

    # Pipeline processing
    metrics = [] # List of (h, c, f, idx)
    junk_phrases = ['click here to order', 'free shipping', 'best price guaranteed']
    
    for i, s in enumerate(norm_docs):
        # Loose wiki gate
        if len(s) < 200: continue
        is_wiki = s.startswith('=')
        has_special = any(x in s for x in [' @-@ ', ' @,@ ', ' @.@ '])
        tokens = s.split()
        rate = sum(1 for t in tokens if t == ',' or t == '.') / len(tokens) if tokens else 0
        if rate >= 0.03: is_wiki = True
        
        if is_wiki or has_special:
            # Calculate h
            bg = get_bigrams(s)
            h = sum(1 for b in bg[:3000] if b in S) / min(len(bg), 3000) if len(bg) > 0 else 0
            
            # Calculate c
            c_count = 0
            c_total = 0
            for j in range(len(tokens)-1):
                if tokens[j] == '.':
                    c_total += 1
                    next_t = tokens[j+1]
                    if next_t and (next_t[0].isupper() or next_t[0].isdigit() or next_t[0] in ['=', '"', "'", '(', '[']):
                        c_count += 1
            c = c_count / c_total if c_total > 0 else 1.0
            
            # Calculate f
            f_words_bg = [b for b in bg if b[0] in F and b[1] in F]
            f = sum(1 for b in f_words_bg if b in S) / len(f_words_bg) if len(f_words_bg) >= 30 else 1.0
            
            # Filter A
            if h >= 0.6 * m:
                doc_orig = docs[deduped_indices[i]]
                ratio = len(zlib.compress(doc_orig.encode('utf-8'))) / len(doc_orig) if len(doc_orig) > 0 else 0
                if ratio >= 0.2 and not any(p in doc_orig.lower() for p in junk_phrases):
                    metrics.append((h, c, f, deduped_indices[i]))

    # Result A
    indices_a = [m[3] for m in metrics]
    
    # Result B
    indices_b = [m[3] for m in metrics if m[1] >= c_thr and m[2] >= f_thr]
    
    # Result C (Title-level near-duplicate)
    groups = {}
    for h, c, f, idx in metrics:
        doc = docs[idx]
        s = doc.strip()
        first_line = s.split('\n')[0]
        if first_line.startswith('=') and len(first_line) < 200:
            key = ' '.join(first_line.lower().split())
        else:
            key = doc # This will be handled by the "is its own group" logic
            # To handle "is its own group", we use a unique id for docs that aren't title-grouped
            key = f"__unique__{idx}"
        
        if key not in groups: groups[key] = []
        groups[key].append({'idx': idx, 'q': h * c * f, 'len': len(doc)})
    
    indices_c = []
    num_title_groups_multi = 0
    for key, items in groups.items():
        if key.startswith("__unique_"):
            indices_c.append(items[0]['idx'])
            continue
        
        num_title_groups_multi += 1
        best_item = max(items, key=lambda x: (x['q'], x['len']))
        # Exception check
        lengths = [x['len'] for x in items]
        shortest, longest = min(lengths), max(lengths)
        if shortest < 0.5 * longest:
            candidates = [x for x in items if x['q'] >= 0.9 * best_item['q']]
            if candidates:
                best_item = max(candidates, key=lambda x: (x['q'], x['len']))
        
        indices_c.append(best_item['idx'])

    # Safety
    chars_a = sum(len(docs[i]) for i in indices_a)
    chars_c = sum(len(docs[i]) for i in indices_c)
    
    if len(indices_c) >= 200 and chars_c >= 0.5 * chars_a:
        final_indices = indices_c
    elif len(indices_b) >= 200 and chars_c < 0.5 * chars_a: # This logic is slightly flawed, re-reading rule
        # "if C has fewer than 200 docs or fewer than 50% of the characters of A, 
        # return B if B meets those same two conditions, else return A."
        # The "those same two conditions" refers to C's conditions: |C|>=200 and chars(C) >= 0.5*chars(A)
        # Wait, "return B if B meets those same two conditions" means |B|>=200 and chars(B) >= 0.5*chars(A)
        # Let's re-verify logic.
        pass
    
    # Refined Safety Logic:
    chars_b = sum(len(docs[i]) for i in indices_b)
    if len(indices_c) >= 200 and chars_c >= 0.5 * chars_a:
        final_indices = indices_c
    elif len(indices_b) >= 200 and chars_b >= 0.5 * chars_a:
        final_indices = indices_b
    else:
        final_indices = indices_a

    # Printing metrics
    print(f"|A|: {len(indices_a)}")
    print(f"|B|: {len(indices_b)}")
    print(f"|C|: {len(indices_c)}")
    print(f"Groups > 1: {num_title_groups_multi}")
    print(f"chars(C)/chars(A): {chars_c/chars_a if chars_a > 0 else 0}")
    print(f"tokens(C)/T: {sum(len(docs[i]) for i in indices_c)/4 / T if T > 0 else 0}")

    rng.shuffle(final_indices)
    return [int(i) for i in final_indices]
