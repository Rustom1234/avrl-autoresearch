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
        for i in range(len(words) - 2):
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            h = hashlib.md5(tri_str.encode('utf-8')).digest()
            trigrams.add(h)
        return trigrams, len(trigrams)

    def get_bigrams(text):
        words = text.lower().split()
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add((words[i], words[i+1]))
        return bigrams

    # 1. Build Bigram set B from target docs
    B = set()
    for doc in target_sample:
        B.update(get_bigrams(doc))
    
    # 2. Calculate r for target docs and find r0
    target_rs = []
    for doc in target_sample:
        tokens = enc.encode(doc)
        if len(tokens) < 10:
            target_rs.append(0.0)
            continue
        # Use first 3000 tokens for r calculation
        sub_tokens = tokens[:3000]
        # Reconstruct words for bigrams from sub_tokens (simplified)
        # To stay efficient and accurate to "word bigrams", we use text segment
        text_part = enc.decode(sub_tokens)
        words = text_part.lower().split()
        if len(words) < 2:
            target_rs.append(0.0)
            continue
        
        count = 0
        hits = 0
        for i in range(len(words) - 1):
            bi = (words[i], words[i+1])
            count += 1
            if bi in B:
                hits += 1
        target_rs.append(hits / count if count > 0 else 0.0)
    
    r0 = min(target_rs) if target_rs else 0.0
    
    # 3. Scoring and Filters
    target_trigrams = set()
    target_m_vals = []
    for doc in target_sample:
        trigs, count = get_trigrams(doc)
        target_trigrams.update(trigs)
        # h = hits / count
        trigs_in_target, _ = get_trigrams(doc)
        hits = sum(1 for t in trigs_in_target if t in target_trigrams)
        target_m_vals.append(hits / len(trigs_in_target) if len(trigs_in_target) > 0 else 0.0)
    
    t_thresh = min(target_m_vals) if target_m_vals else 0.0
    if len(target_m_vals) > 0:
        target_m_vals.sort()
        t_thresh = target_m_vals[int(0.05 * len(target_m_vals))]

    valid_docs_data = []
    
    for i, doc in enumerate(docs):
        # Alphabetic check
        tokens = enc.encode(doc)
        if len(tokens) < 10: continue
        
        # Check first 3000 tokens
        sub_tokens = tokens[:3000]
        sub_text = enc.decode(sub_tokens)
        words = sub_text.lower().split()
        
        # Alphabetic/Space rate
        alpha_count = 0
        total_chars = 0
        for char in sub_text:
            total_chars += 1
            if 'a' <= char <= 'z' or char == ' ':
                alpha_count += 1
        
        # Windowed check
        win_passed = True
        win_size = 500
        for start in range(0, len(sub_text), win_size):
            win = sub_text[start:start+win_size]
            if len(win) == 0: break
            w_alpha = sum(1 for c in win if 'a' <= c <= 'z' or c == ' ')
            if w_alpha / len(win) < 0.55:
                win_passed = False
                break
        
        if alpha_count / total_chars < 0.50:
            continue

        # Bigram r calculation
        r_val = 0.0
        if len(words) >= 2:
            bi_count = 0
            bi_hits = 0
            for j in range(len(words)-1):
                bi = (words[j], words[j+1])
                bi_count += 1
                if bi in B:
                    bi_hits += 1
            r_val = bi_hits / bi_count if bi_count > 0 else 0.0
        
        # Damage conditions
        is_damaged = (r_val < 0.6 * r0)
        
        # Trigram h calculation
        trigs, trig_count = get_trigrams(doc)
        h = sum(1 for tr in trigs if tr in target_trigrams) / trig_count if trig_count > 0 else 0.0
        
        # Filters
        junk_markers = {'coupon', 'discount code', 'free shipping', 'buy now', 'Order cannes', 'Get hazard'}
        is_junk = sum(1 for m in junk_markers if m in doc.lower()) >= 3
        
        if is_junk: continue
        if is_damaged or (h < 0.4 * t_thresh and trig_count < 100):
            continue
            
        # Unique tokens check
        unique_tokens = len(set(tokens))
        if unique_tokens < 150:
            continue
            
        valid_docs_data.append({
            'idx': i, 
            'h': h, 
            'tokens': len(tokens), 
            'r': r_val,
            'doc': doc
        })

    # 4. Selection logic
    scored_docs = []
    for d in valid_docs_data:
        # Use h with noise
        scored_docs.append({'idx': d['idx'], 'h': d['h'], 'tokens': d['tokens'], 'original_h': d['h']})
    
    if len(scored_docs) > 1:
        hs = [d['h'] for d in scored_docs]
        std_h = np.std(hs)
        for d in scored_docs:
            d['score'] = d['h'] + rng.uniform(-0.1 * std_h, 0.1 * std_h)
    else:
        for d in scored_docs:
            d['score'] = d['h']
            
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    high_score_indices = [d['idx'] for d in scored_docs if d['original_h'] >= 0.5 * t_thresh]
    high_score_tokens = sum(len(enc.encode(docs[i])) for i in high_score_indices)
    
    final_indices = []
    if high_score_tokens >= 0.7 * T:
        final_indices = high_score_indices[:]
        rng.shuffle(final_indices)
    else:
        current_indices = high_score_indices[:]
        current_tokens = high_score_tokens
        for d in scored_docs:
            if d['idx'] not in current_indices:
                current_indices.append(d['idx'])
                current_tokens += d['tokens']
                if current_tokens >= 0.7 * T:
                    break
        rng.shuffle(current_indices)
        final_indices = current_indices

    # Final T check and repetition
    total_unique_tokens = sum(len(enc.encode(docs[i])) for i in final_indices)
    
    if total_unique_tokens == 0:
        return []
    
    if total_unique_tokens < 0.5 * T:
        # Return only this set as per instruction (it's the "surviving clean set")
        rng.shuffle(final_indices)
        return final_indices

    # Order to fulfill "repeat the kept list"
    res = []
    while sum(len(enc.encode(docs[i])) for i in res) < T:
        res.extend(final_indices)
    
    return res
