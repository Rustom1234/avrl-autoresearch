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

    def get_bigrams(text):
        words = text.lower().split()
        if len(words) < 2:
            return set(), 0
        bigrams = set()
        for i in range(len(words) - 1):
            bi = f"{words[i]} {words[i+1]}"
            bigrams.add(bi)
        return bigrams, len(bigrams)

    def get_trigrams(text):
        words = text.lower().split()
        if len(words) < 3:
            return set(), 0
        trigrams = set()
        for i in range(len(words) - 2):
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            h = hashlib.md5(tri_str.encode('utf-8')).digest()
            trigrams.add(h)
        return trigrams, len(trigrams)

    # 1. Build Reference Sets
    H = set()
    R = {}
    for doc in target_sample:
        bi_set, _ = get_bigrams(doc)
        H.update(bi_set)
        words = doc.lower().split()
        for i in range(len(words) - 1):
            bi = f"{words[i]} {words[i+1]}"
            R[bi] = R.get(bi, 0) + 1
            
    sorted_r = sorted(R.items(), key=lambda x: x[1], reverse=True)[:3000]
    R_set = set(item[0] for item in sorted_r)
    H_plus_R = H.union(R_set)

    def calc_q(doc):
        bi_set, count = get_bigrams(doc)
        if count == 0: return 0.0, 0.0
        h_matches = sum(1 for b in bi_set if b in H)
        q = h_matches / count
        q1 = sum(1 for b in bi_set if b in H_plus_R) / count
        return q, q1

    target_q_vals = []
    for doc in target_sample:
        q, q1 = calc_q(doc)
        target_q_vals.append(q)
        target_q_vals.append(q1)

    q0 = min(target_q_vals) if target_q_vals else 0.0
    med = np.median(target_q_vals) if target_q_vals else 0.0

    # Trigram scoring
    target_trigrams = set()
    for doc in target_sample:
        trigs, _ = get_trigrams(doc)
        target_trigrams.update(trigs)
    
    m_vals = []
    for doc in target_sample:
        trigs, count = get_trigrams(doc)
        if count > 0:
            m_vals.append(sum(1 for t in trigs if t in target_trigrams) / count)
        else:
            m_vals.append(0.0)
    
    t_val = sorted(m_vals)[int(0.05 * len(m_vals))] if m_vals else 0.0

    # Scoring Pool
    scored_docs = []
    junk_markers = {'coupon', 'discount code', 'free shipping', 'buy now', 'order cannes', 'get hazard'}
    
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        if sum(1 for marker in junk_markers if marker in doc_lower) >= 3:
            continue
        
        trigs, count = get_trigrams(doc)
        h = sum(1 for t in trigs if t in target_trigrams) / count if count > 0 else 0.0
        
        if h < 0.4 * t_val and count < 100:
            continue
            
        q, q1 = calc_q(doc)
        if (q < 0.75 * q0 and count >= 150) or (q1 < 0.6 * med and count >= 100):
            continue
            
        if len(set(doc)) < 5000:
            continue
            
        scored_docs.append({'idx': i, 'h': h, 'q1': q1, 'tokens': len(enc.encode(doc))})

    # Filtering and Selection
    clean_set = [d for d in scored_docs if d['q1'] >= 0.6 * med]
    clean_set.sort(key=lambda x: x['h'], reverse=True)
    
    final_indices = []
    curr_tokens = 0
    
    for d in clean_set:
        if curr_tokens + d['tokens'] <= 0.7 * T:
            final_indices.append(d['idx'])
            curr_tokens += d['tokens']
        else:
            break
            
    remaining = [d for d in scored_docs if d['idx'] not in final_indices]
    remaining.sort(key=lambda x: x['h'], reverse=True)
    for d in remaining:
        if curr_tokens + d['tokens'] <= 1.025 * T:
            final_indices.append(d['idx'])
            curr_tokens += d['tokens']
        else:
            break

    if not final_indices:
        return []

    rng.shuffle(final_indices)
    return final_indices
