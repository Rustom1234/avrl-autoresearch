import os
import random
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def get_tokens(d):
        try:
            return len(enc.encode_ordinary(str(d)))
        except:
            return len(str(d)) // 4

    doc_data = []
    for i, d in enumerate(docs):
        t_count = get_tokens(d)
        is_junk = False
        if len(d) < 10 or (len(d) > 100 and d.count('\n') == 0 and '.' not in d):
            is_junk = True
        
        is_wiki = False
        if d.startswith('= '):
            first_3k = d[:3000]
            count = 0
            for char in [' ', ',', '.', '@', '-']:
                count += first_3k.count(char)
            if count >= 5:
                is_wiki = True
        
        doc_data.append({
            'id': i,
            'tokens': t_count,
            'is_junk': is_junk,
            'is_wiki': is_wiki,
        })

    tier_a = []
    tier_b = []
    tier_d = []

    for item in doc_data:
        if item['is_junk']:
            tier_d.append(item)
        elif item['is_wiki']:
            tier_a.append(item)
        else:
            tier_b.append(item)

    target_texts = [str(s) for s in target_sample]
    vec = HashingVectorizer(ngram_range=(1, 2)).fit(target_texts)
    target_matrix = vec.transform(target_texts).toarray()
    centroid = np.mean(target_matrix, axis=0).reshape(1, -1)

    def get_sim(text):
        v = vec.transform([text]).toarray()
        return cosine_similarity(v, centroid)[0][0]

    for item in tier_b:
        item['sim'] = get_sim(docs[item['id']])
    
    tier_b.sort(key=lambda x: x['sim'], reverse=True)

    target_limit = 1.01 * T
    tokens_a = sum(item['tokens'] for item in tier_a)

    if tokens_a >= target_limit:
        s_a_docs = tier_a
        total_s_a_tokens = tokens_a
        half_tokens = total_s_a_tokens / 2
        a_early, a_late = [], []
        running_a_tokens = 0
        for item in s_a_docs:
            if running_a_tokens + item['tokens'] <= half_tokens:
                a_early.append(item)
                running_a_tokens += item['tokens']
            else:
                a_late.append(item)
        if not a_early and a_late:
            a_early.append(a_late.pop(0))
        s_non_a = [item for item in doc_data if not (item['is_wiki'] and not item['is_junk'])]
        phase1 = a_early + s_non_a
        rng.shuffle(phase1)
        phase2 = a_late
        rng.shuffle(phase2)
        return [item['id'] for item in phase1] + [item['id'] for item in phase2]
    else:
        rem = target_limit - tokens_a
        
        a_copy = []
        current_a_tokens = 0
        
        # Max 3 passes total (Initial + 2 extra)
        for _ in range(3):
            rng.shuffle(tier_a)
            for item in tier_a:
                if current_a_tokens + item['tokens'] <= rem:
                    a_copy.append(item)
                    current_a_tokens += item['tokens']
        
        forbidden = ['discount code', 'coupon', 'click here to order', 'best price guaranteed', 'free shipping']
        filtered_b = []
        for item in tier_b:
            txt = docs[item['id']].lower()
            if not any(f in txt for f in forbidden):
                filtered_b.append(item)
        
        filler = []
        current_filler_tokens = 0
        combined_filler_pool = filtered_b + tier_d
        
        for item in combined_filler_pool:
            if current_filler_tokens + item['tokens'] <= (rem - current_a_tokens):
                filler.append(item)
                current_filler_tokens += item['tokens']
        
        phase1 = a_copy + filler
        rng.shuffle(phase1)
        
        phase2 = list(tier_a)
        rng.shuffle(phase2)
        
        res_ids = [item['id'] for item in phase1] + [item['id'] for item in phase2]
        total_res_tokens = sum(doc_data[i]['tokens'] for i in res_ids)
        
        if total_res_tokens < target_limit:
            needed = target_limit - total_res_tokens
            # Extract items not already in the list to fill from front
            used_ids = set(res_ids)
            extra_pool = []
            # Priority: filtered_b then tier_d
            for item in filtered_b:
                if item['id'] not in used_ids: extra_pool.append(item)
            for item in tier_d:
                if item['id'] not in used_ids: extra_pool.append(item)
                
            for item in extra_pool:
                if needed <= 0: break
                res_ids.insert(0, item['id'])
                needed -= item['tokens']
                
        return res_ids
