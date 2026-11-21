import os
import random
import hashlib
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")

    def get_trigrams(text):
        words = text.lower().split()
        if len(words) < 3:
            return set(), 0
        trigrams = set()
        for i in range(len(words) - 3):
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            trigrams.add(hashlib.md5(tri_str.encode('utf-8')).digest())
        return trigrams, len(trigrams)

    def get_bigram_score(text, bigram_set):
        s = text.lower()
        num = 0
        den = 0
        for i in range(len(s) - 1):
            pair = s[i:i+2]
            if pair in bigram_set:
                num += 1
            if s[i].isalpha() and s[i+1].isalpha():
                den += 1
        if den == 0: return 1.0 if num == 0 else 0.0
        return num / den

    # 1. Build Sets
    target_trigrams_set = set()
    target_bigrams_set = set()
    target_bigrams_counts = {}
    
    for doc in target_sample:
        tokens = enc.encode(doc)
        doc_subset = enc.decode(tokens[:2000])
        
        # Trigrams
        trigs, _ = get_trigrams(doc_subset)
        target_trigrams_set.update(trigs)
        
        # Bigrams
        s = doc_subset.lower()
        for i in range(len(s) - 1):
            pair = s[i:i+2]
            target_bigrams_set.add(pair)
            target_bigrams_counts[pair] = target_bigrams_counts.get(pair, 0) + 1
            
    # 2. Calculate metrics
    target_d_values = []
    target_d1_vals = []
    target_d2_vals = []
    
    sorted_counts = sorted(target_bigrams_counts.items(), key=lambda x: x[1], reverse=True)
    D1 = set(target_bigrams_set)
    for i in range(min(300, len(sorted_counts))):
        D1.add(sorted_counts[i][0])
    D1.update(['or ', ', ', '. ', ' @', ' ,', ' .'])
    
    D2 = set(target_bigrams_set)
    for i in range(min(1500, len(sorted_counts))):
        D2.add(sorted_counts[i][0])
    D2.update(['or ', ', ', '. ', ' @', ' ,', ' .'])

    for doc in target_sample:
        tokens = enc.encode(doc)
        doc_subset = enc.decode(tokens[:2000])
        s = doc_subset.lower()
        
        d0_val = get_bigram_score(doc_subset, target_bigrams_set)
        d1_val = get_bigram_score(doc_subset, D1)
        d2_val = get_bigram_score(doc_subset, D2)
        
        target_d_values.append(d0_val)
        target_d1_vals.append(d1_val)
        target_d2_vals.append(d2_val)
    
    d0_min = min(target_d_values) if target_d_values else 0.0
    med_d = np.median(target_d_values) if target_d_values else 0.0
    mean_d1 = np.mean(target_d1_vals) if target_d1_vals else 0.0
    mean_d2 = np.mean(target_d2_vals) if target_d2_vals else 0.0

    def is_secondarily_damaged(doc):
        tok_len = len(enc.encode(doc))
        if tok_len < 300:
            return False
        
        tokens = enc.encode(doc)
        doc_subset = enc.decode(tokens[:2000])
        s = doc_subset.lower()
        
        d0_val = get_bigram_score(doc_subset, target_bigrams_set)
        d1_val = get_bigram_score(doc_subset, D1)
        d2_val = get_bigram_score(doc_subset, D2)
        
        if (d0_val < d0_min * 0.5) or (d1_val < mean_d1 * 0.7) or (d2_val < mean_d2 * 0.7):
            return True
        
        mean_d = (d0_val + d1_val + d2_val) / 3.0
        if mean_d <= 0.4 * med_d:
            return True
        return False

    # 3. Filtering
    surviving_docs = []
    for i, doc in enumerate(docs):
        if is_secondarily_damaged(doc):
            continue
        
        trigs, count = get_trigrams(doc)
        h = sum(1 for tr in trigs if tr in target_trigrams_set) / count if count > 0 else 0.0
        surviving_docs.append({'idx': i, 'h': h, 'tokens': len(enc.encode(doc))})

    # 4. Scoring
    target_h_vals = []
    for doc in target_sample:
        trigs, count = get_trigrams(doc)
        target_h_vals.append(sum(1 for tr in trigs if tr in target_trigrams_set) / count if count > 0 else 0.0)
    
    target_h_vals.sort()
    t_thresh = target_h_vals[int(0.05 * len(target_h_vals))] if target_h_vals else 0.0
    
    scored_docs = []
    for d in surviving_docs:
        scored_docs.append(d)
    
    if len(scored_docs) > 1:
        hs = [d['h'] for d in scored_docs]
        std_h = np.std(hs)
        for d in scored_docs:
            d['score'] = d['h'] + rng.uniform(-0.1 * std_h, 0.1 * std_h)
    else:
        for d in scored_docs:
            d['score'] = d['h']

    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    high_score_indices = [d['idx'] for d in scored_docs if d['h'] >= 0.5 * t_thresh]
    
    high_score_tokens = sum(len(enc.encode(docs[i])) for i in high_score_indices)
    
    final_indices = []
    if high_score_tokens >= 0.7 * T:
        final_indices = high_score_indices[:]
    else:
        current_indices = high_score_indices[:]
        current_tokens = high_score_tokens
        for d in scored_docs:
            if d['idx'] not in current_indices:
                current_indices.append(d['idx'])
                current_tokens += d['tokens']
                if current_tokens >= 0.7 * T:
                    break
        final_indices = current_indices

    if not final_indices:
        return []

    rng.shuffle(final_indices)
    res_final = final_indices[:]
    total_tokens = sum(len(enc.encode(docs[i])) for i in final_indices)
    
    while total_tokens < T:
        res_final.extend(final_indices)
        total_tokens += sum(len(enc.encode(docs[i])) for i in final_indices)
        
    return res_final
