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

    # Pre-calculate metadata
    doc_data = []
    for i, d in enumerate(docs):
        t_count = get_tokens(d)
        
        # Junk flag
        is_junk = False
        if len(d) < 10 or (len(d) > 100 and d.count('\n') == 0 and '.' not in d):
            is_junk = True
        
        # Wiki flag
        is_wiki = False
        if d.startswith('= '):
            first_3k = d[:3000]
            count = 0
            for char in [' ', ',', '.', '@', '-']:
                count += first_3k.count(char)
            if count >= 5:
                is_wiki = True
        
        # p_clean (simplified proxy as labels are unavailable)
        p_clean = 0.5
        
        doc_data.append({
            'id': i,
            'tokens': t_count,
            'is_junk': is_junk,
            'is_wiki': is_wiki,
            'p_clean': p_clean
        })

    # Filter tiers
    tier_a = [] # Clean wiki
    tier_b = [] # Clean non-wiki
    tier_d = [] # Junk/Rest

    for item in doc_data:
        if item['is_junk']:
            tier_d.append(item)
        elif item['is_wiki']:
            tier_a.append(item)
        else:
            tier_b.append(item)
            
    # Similarity for Tier B
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
    
    # Selection
    selected_indices = []
    total_tokens = 0
    target_limit = 1.01 * T
    
    S = []
    a_tokens_total_pool = sum(item['tokens'] for item in tier_a)
    
    if a_tokens_total_pool >= target_limit:
        tier_a.sort(key=lambda x: x['p_clean'], reverse=True)
        current_s_tokens = 0
        for item in tier_a:
            if current_s_tokens + item['tokens'] <= target_limit * 1.01:
                S.append(item)
                current_s_tokens += item['tokens']
    else:
        S.extend(tier_a)
        current_s_tokens = a_tokens_total_pool
        for item in tier_b:
            if current_s_tokens + item['tokens'] <= target_limit * 1.01:
                S.append(item)
                current_s_tokens += item['tokens']
        for item in tier_d:
            if current_s_tokens + item['tokens'] <= target_limit * 1.01:
                S.append(item)
                current_s_tokens += item['tokens']

    # Ordering
    s_a_docs = [item for item in S if item['is_wiki'] and not item['is_junk']]
    total_s_a_tokens = sum(item['tokens'] for item in s_a_docs)
    
    half_tokens = total_s_a_tokens / 2
    a_early = []
    a_late = []
    running_a_tokens = 0
    for item in s_a_docs:
        if running_a_tokens + item['tokens'] <= half_tokens:
            a_early.append(item)
            running_a_tokens += item['tokens']
        else:
            a_late.append(item)
            
    if not a_early and a_late:
        a_early.append(a_late.pop(0))

    # Phase 1: A_early + all non-A docs (B, D)
    s_non_a = [item for item in S if not (item['is_wiki'] and not item['is_junk'])]
    phase1 = a_early + s_non_a
    rng.shuffle(phase1)
    
    # Phase 2: A_late
    phase2 = a_late
    rng.shuffle(phase2)
    
    final_selection = [item['id'] for item in phase1] + [item['id'] for item in phase2]
    return final_selection
