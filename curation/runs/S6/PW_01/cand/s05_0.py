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

    def get_bigrams(text, limit_tokens=3000):
        tokens = enc.encode(text)
        if len(tokens) < 2:
            return set(), 0
        # Use first 3000 tokens for bigram analysis
        subset = tokens[:limit_tokens]
        words = enc.decode(subset).lower().split()
        if len(words) < 2:
            return set(), 0
        bigrams = set()
        for i in range(len(words) - 1):
            bi_str = f"{words[i]} {words[i+1]}".lower()
            bigrams.add(bi_str)
        return bigrams, len(bigrams)

    # Pass 1: Trigram collection and setup
    target_trigrams = set()
    target_doc_trigrams = []
    # Pass 2: Bigram collection
    target_bigrams = set()
    target_doc_bigrams = []

    for doc in target_sample:
        trigs, count = get_trigrams(doc)
        target_doc_trigrams.append(count)
        target_trigrams.update(trigs)
        
        bi_set, bi_count = get_bigrams(doc)
        target_doc_bigrams.append(bi_count)
        target_bigrams.update(bi_set)

    def calc_h(doc):
        trigs, count = get_trigrams(doc)
        if count == 0: return 0.0
        hits = sum(1 for t in trigs if t in target_trigrams)
        return hits / count

    m_vals = [calc_h(doc) for doc in target_sample]
    m_vals.sort()
    t = m_vals[int(0.05 * len(m_vals))] if m_vals else 0.0

    # Scoring and filtering
    scored_docs = []
    junk_markers = {'coupon', 'discount code', 'free shipping', 'buy now', 'Order cannes', 'Get hazard'}
    
    for i, doc in enumerate(docs):
        if sum(1 for marker in junk_markers if marker in doc.lower()) >= 3:
            continue
        
        trigs, count = get_trigrams(doc)
        h = sum(1 for tr in trigs if tr in target_trigrams) / count if count > 0 else 0.0
        
        # Damage check 1
        if h < 0.4 * t and count < 100:
            continue
        
        # Damage check 2 (Bigrams)
        unique_tokens = len(set(enc.encode(doc)))
        if unique_tokens >= 100:
            bi_set, bi_count = get_bigrams(doc)
            if bi_count > 0:
                hits = sum(1 for b in bi_set if b in target_bigrams)
                r = hits / bi_count
                r0 = min(target_doc_bigrams) if target_doc_bigrams else 0.0
                if r < 0.5 * r0:
                    continue
                    
        scored_docs.append({'idx': i, 'h': h, 'tokens': len(enc.encode(doc))})

    if len(scored_docs) > 1:
        hs = [d['h'] for d in scored_docs]
        std_h = np.std(hs)
        for d in scored_docs:
            d['score'] = d['h'] + rng.uniform(-0.1 * std_h, 0.1 * std_h)
    else:
        for d in scored_docs:
            d['score'] = d['h']

    scored_docs.sort(key=lambda x: x['score'], reverse=True)

    # Selection Logic
    high_score_indices = [d['idx'] for d in scored_docs if d['h'] >= 0.5 * t]
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

    if sum(len(enc.encode(docs[i])) for i in final_indices) < 0.5 * T:
        rng.shuffle(final_indices)
        return final_indices

    # Final constraints
    rng.shuffle(final_indices)
    res_final = final_indices[:]
    total_len = sum(len(enc.encode(docs[i])) for i in res_final)
    while total_len < T:
        res_final.extend(final_indices)
        total_len = sum(len(enc.encode(docs[i])) for i in res_final)
    
    return res_final
