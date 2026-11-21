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

    # 1. Build Bigram Sets
    H = set()
    R = {} # Using dict to count frequency for R
    target_q_vals = []
    
    for doc in target_sample:
        bi_set, _ = get_bigrams(doc)
        H.update(bi_set)
        # Count for R
        words = doc.lower().split()
        for i in range(len(words) - 1):
            bi = f"{words[i]} {words[i+1]}"
            R[bi] = R.get(bi, 0) + 1
            
    # R = 3000 most frequent
    sorted_r = sorted(R.items(), key=lambda x: x[1], reverse=True)[:3000]
    R_set = set(item[0] for item in sorted_r)
    H_plus_R = H.union(R_set)

    def calc_q(doc):
        bi_set, count = get_bigrams(doc)
        if count == 0: return 0.0
        h_matches = sum(1 for b in bi_set if b in H)
        q = h_matches / count
        q1 = sum(1 for b in bi_set if b in H_plus_R) / count
        return q, q1

    for doc in target_sample:
        q, q1 = calc_q(doc)
        target_q_vals.append(q)
        target_q_vals.append(q1)

    q0 = min(target_q_vals) if target_q_vals else 0.0
    med = np.median(target_q_vals) if target_q_vals else 0.0

    # Score and Filter Pool
    scored_docs = []
    junk_markers = {'coupon', 'discount code', 'free shipping', 'buy now', 'Order cannes', 'Get hazard'}
    
    # Trigram check (First pass)
    target_trigrams = set()
    for doc in target_sample:
        trigs, _ = get_trigrams(doc)
        target_trigrams.update(trigs)
    
    t_threshold = sorted([h_val for doc in target_sample for h_val, _ in [(sum(1 for t in get_trigrams(doc)[0] if t in target_trigrams) / get_trigrams(doc)[1], 0) for doc in target_sample]])][int(0.05 * 200)] if target_sample else 0.0
    # Correction: let's just use a simple 0.4 * t_threshold logic as before
    # But wait, the instructions say "retrain selection on surviving docs only".
    
    # Pre-calc T_val (baseline for h)
    def get_h(doc):
        trigs, count = get_trigrams(doc)
        if count == 0: return 0.0
        return sum(1 for t in trigs if t in target_trigrams) / count

    m_vals = [get_h(doc) for doc in target_sample]
    t = sorted(m_vals)[int(0.05 * len(m_vals))] if m_vals else 0.0

    for i, doc in enumerate(docs):
        if sum(1 for marker in junk_markers if marker in doc.lower()) >= 3:
            continue
        
        h = get_h(doc)
        trigs, count = get_trigrams(doc)
        
        if h < 0.4 * t and count < 100:
            continue
            
        q, q1 = calc_q(doc)
        
        # Secondary Damage
        # q < 0.75 * q0 and holds >= 150 unique bigrams
        # OR q1 < 0.6 * med and holds >= 100 unique bigrams
        if (q < 0.75 * q0 and count >= 150) or (q1 < 0.6 * med and count >= 100):
            continue
            
        # Final Guard: unique tokens < 5000 chars
        unique_chars = len(set(doc))
        if unique_chars < 5000:
            continue
            
        scored_docs.append({'idx': i, 'h': h, 'q1': q1, 'tokens': len(enc.encode(doc))})

    # Retrain selection
    # q1 >= 0.6 * med set
    clean_set = [d for d in scored_docs if d['q1'] >= 0.6 * med]
    clean_set.sort(key=lambda x: x['h'], reverse=True)
    
    final_indices = []
    curr_tokens = 0
    
    # Up to 0.7 T
    for d in clean_set:
        if curr_tokens + d['tokens'] <= 0.7 * T:
            final_indices.append(d['idx'])
            curr_tokens += d['tokens']
        else:
            break
            
    # Next best clean docs up to 1.025 T
    remaining = [d for d in scored_docs if d['idx'] not in final_indices]
    remaining.sort(key=lambda x: x['h'], reverse=True)
    for d in remaining:
        if curr_tokens + d['tokens'] <= 1.025 * T:
            final_indices.append(d['idx'])
            curr_tokens += d['tokens']
        else:
            break

    # 0.7 T Hard Fallback
    if curr_tokens < 0.7 * T:
        # Keep high scores
        high_scores = [d for d in scored_docs if d['h'] >= 0.5 * t]
        high_score_tokens = sum(len(enc.encode(docs[d['idx']])) for d in high_scores)
        if high_score_tokens >= 0.7 * T:
            final_indices = [d['idx'] for d in high_scores]
        else:
            # Fill from sorted scored_docs
            temp_indices = []
            temp_tokens = 0
            for d in scored_docs:
                if d['idx'] not in temp_indices:
                    temp_indices.append(d['idx'])
                    temp_tokens += d['tokens']
                    if temp_tokens >= 0.7 * T:
                        break
            final_indices = temp_indices

    if not final_indices:
        return []

    # Final order and repeats
    rng.shuffle(final_indices)
    res_final = []
    # The prompt says "Return the selected indices in seeded random order"
    # And "If the list holds fewer than T tokens it is cycled".
    # "Repeat the kept list; do not bring in unselected"
    
    # Just return the shuffled list once; the trainer cycles it. 
    # To be safe on "repeat the kept list", we can repeat it manually.
    # However, "return the selected indices in seeded random order" is the primary 
    # instruction for the list content.
    
    # Let's just return the shuffled indices.
    return final_indices
