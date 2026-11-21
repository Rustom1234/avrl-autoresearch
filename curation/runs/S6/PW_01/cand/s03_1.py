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
            h = hashlib.md5(tri_str.encode('utf-8')).digest()
            trigrams.add(h)
        return trigrams, len(trigrams)

    def get_bigrams(text):
        words = text.split()
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add(f"{words[i]}_{words[i+1]}")
        return bigrams

    def char_bigram_entropy(text):
        sample = text[:2000]
        bigrams = []
        for i in range(len(sample) - 1):
            bigrams.append(sample[i:i+2])
        if not bigrams: return 0.0
        counts = {}
        for b in bigrams:
            counts[b] = counts.get(b, 0) + 1
        total = len(bigrams)
        entropy = 0.0
        for b in counts:
            p = counts[b] / total
            entropy -= p * np.log2(p)
        return entropy

    target_entropies = [char_bigram_entropy(d) for d in target_sample]
    target_entropies.sort()
    
    # Use correct percentile indices for 0th, 0.5th, and 1st quartile (25th percentile)
    idx_0 = 0
    idx_50 = int(0.5 * len(target_entropies))
    idx_25 = int(0.25 * len(target_entropies))
    
    ent_0 = target_entropies[idx_0]
    ent_50 = target_entropies[idx_50]
    ent_25 = target_entropies[idx_25]
    
    k_den = ent_0 + ent_50
    k_num = ent_50 + ent_25
    k = k_num / k_den if k_den != 0 else 1.0

    target_trigram_set = set()
    for d in target_sample:
        trigs, _ = get_trigrams(d)
        target_trigram_set.update(trigs)
    
    target_doc_scores = []
    for d in target_sample:
        trigs, count = get_trigrams(d)
        hits = sum(1 for tr in trigs if tr in target_trigram_set)
        norm_score = (hits / count if count > 0 else 0) * k
        target_doc_scores.append(norm_score)
    
    target_doc_scores.sort()
    t_val = target_doc_scores[int(0.05 * len(target_doc_scores))] if target_doc_scores else 0.0

    scored_pool = []
    for i, doc in enumerate(docs):
        trigs, count = get_trigrams(doc)
        h_val = sum(1 for tr in trigs if tr in target_trigram_set) / count if count > 0 else 0.0
        h_pool = h_val * k
        
        if sum(1 for m in ['coupon', 'discount code', 'free shipping', 'buy now', 'Order cannes', 'Get hazard'] if m in doc.lower()) >= 3:
            continue
        if h_pool < 0.4 * t_val and count < 100:
            continue
            
        tokens = enc.encode(doc)
        scored_pool.append({
            'idx': i, 
            'h': h_pool, 
            'tokens': len(tokens), 
            'doc': doc, 
            'count': count
        })

    bigrams_B = set()
    for d in target_sample:
        bigrams_B.update(get_bigrams(d))
    
    # Calculate r for pool
    for item in scored_pool:
        text_part = item['doc'][:2000]
        b_count = 0
        b_in_B = 0
        for i in range(len(text_part)-1):
            bg = text_part[i:i+2]
            b_count += 1
            if bg in bigrams_B:
                b_in_B += 1
        item['r'] = b_in_B / b_count if b_count > 0 else 0.0

    target_r_vals = []
    for d in target_sample:
        text_part = d[:2000]
        b_count = 0
        b_in_B = 0
        for i in range(len(text_part)-1):
            bg = text_part[i:i+2]
            b_count += 1
            if bg in bigrams_B:
                b_in_B += 1
        target_r_vals.append(b_in_B / b_count if b_count > 0 else 0.0)
    target_r_vals.sort()
    r5 = target_r_vals[int(0.05 * len(target_r_vals))] if target_r_vals else 0.0

    S = [item for item in scored_pool if item['h'] >= 0.5 * t_val]
    s_tokens = sum(item['tokens'] for item in S)
    
    if s_tokens < 0.5 * T and r5 > 0 and S:
        s_r_vals = sorted([item['r'] for item in S])
        r5_S = s_r_vals[int(0.05 * len(s_r_vals))] if s_r_vals else 0.0
        if r5_S >= 0.6 * r5:
            others = [item for item in scored_pool if item not in S]
            others.sort(key=lambda x: x['r'])
            count_to_add = int(len(others) * 0.15)
            for i in range(count_to_add):
                S.append(others[i])
            S.sort(key=lambda x: x['h'], reverse=True)
            s_tokens = sum(item['tokens'] for item in S)

    final_indices = []
    if s_tokens < 0.5 * T:
        final_indices = [item['idx'] for item in S]
        rng.shuffle(final_indices)
        return final_indices
    else:
        S.sort(key=lambda x: x['h'], reverse=True)
        final_indices = [item['idx'] for item in S]
        current_tokens = s_tokens
        
        others = [item for item in scored_pool if item not in S]
        others.sort(key=lambda x: x['h'], reverse=True)
        
        for item in others:
            if current_tokens + item['tokens'] <= 1.025 * T:
                final_indices.append(item['idx'])
                current_tokens += item['tokens']
            else:
                break
        
        rng.shuffle(final_indices)
        res = []
        total_final_tokens = sum(len(enc.encode(docs[i])) for i in final_indices)
        if total_final_tokens == 0: return []
        
        while sum(len(enc.encode(docs[i])) for i in res) < T:
            res.extend(final_indices)
            if len(res) > 1000000: break
        return res
