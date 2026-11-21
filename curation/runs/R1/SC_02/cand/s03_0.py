import os
import random
import re
import collections
import math
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken

def select(docs: List[str], target_sample: List[str]) -> List[int]:
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenizer setup with fallback
    try:
        enc = tiktoken.get_encoding("gpt2")
        def get_tokens(text_list):
            return enc.encode_ordinary_batch(text_list, num_threads=1)
    except Exception:
        def get_tokens(text_list):
            return [list(t) for t in text_list] # Dummy but prevents crash
        
    def count_tokens(texts):
        try:
            tokens = get_tokens(texts)
            return [len(t) for t in tokens]
        except Exception:
            return [len(t) // 4 for t in texts]

    # 1. Dedupe
    unique_docs = []
    seen_hashes = set()
    for d in docs:
        normalized = " ".join(d.lower().split())
        h = hash(normalized)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)
    
    # Pre-calculate token counts for the pool
    doc_token_counts = count_tokens(unique_docs)
    
    # 2. Scoring
    scores = []
    
    # TF-IDF Prep
    # Use first 4000 chars for TF-IDF
    tfidf_texts = [d[:4000] for d in unique_docs]
    tfidf_texts.extend([d[:4000] for d in target_sample])
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), 
        sublinear_tf=True, 
        min_df=2, 
        token_pattern=r'\w+'
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(tfidf_texts)
        pool_vectors = tfidf_matrix[:len(unique_docs)]
        target_vectors = tfidf_matrix[len(unique_docs):]
        
        # Centroid of target
        if target_vectors.shape[0] > 0:
            centroid = np.asarray(target_vectors.mean(axis=0))
        else:
            centroid = np.zeros(tfidf_matrix.shape[1])
            
        def get_cosine_sim(vec):
            norm_vec = vec.toarray().flatten()
            norm_centroid = centroid
            mag = math.sqrt(np.sum(norm_vec**2))
            mag_c = math.sqrt(np.sum(norm_centroid**2))
            if mag == 0 or mag_c == 0: return 0.0
            return np.dot(norm_vec, norm_centroid) / (mag * mag_c)

        # Score Domain (b) - Cosine Sim
        sims = np.array([get_cosine_sim(pool_vectors[i]) for i in range(len(unique_docs))])
        if len(sims) > 0:
            s_min, s_max = sims.min(), sims.max()
            if s_max > s_min:
                domain_scores = (sims - s_min) / (s_max - s_min)
            else:
                domain_scores = np.ones(len(sims)) * 0.5
        else:
            domain_scores = np.zeros(len(unique_docs))
            
    except:
        domain_scores = np.zeros(len(unique_docs))

    # Bigram Prep (c)
    target_bigrams = set()
    for d in target_sample:
        words = d.lower().split()
        bigrams = list(zip(words, words[1:]))
        target_bigrams.update(bigrams)
    
    pool_bigrams = []
    for d in unique_docs:
        words = d.lower().split()
        pool_bigrams.append(list(zip(words, words[1:])))
    
    # Simple percentile-like approach for bigrams
    # Since we need percentile rank over the pool
    all_pool_bigrams = [bg for b_list in pool_bigrams for bg in b_list]
    
    # Scoring loop
    final_scores = []
    for i, d in enumerate(unique_docs):
        score = 0.0
        
        # (a) Format
        if d.startswith('= '):
            content = d[:3000]
            markers = [' , ', ' . ', ' @-@ ']
            count = sum(content.count(m) for m in markers)
            if count >= 5:
                score += 2.0
        
        # (b) Domain
        score += float(domain_scores[i])
        
        # (c) Word Order
        words = d.lower().split()
        doc_bigrams = list(zip(words, words[1:]))[:500]
        if len(doc_bigrams) > 0:
            matches = sum(1 for bg in doc_bigrams if bg in target_bigrams)
            # Simple percentile approximation: matches / total_possible
            # We'll use a density score scaled by 1.5
            score += (matches / len(doc_bigrams)) * 1.5
            
        # (d) Junk
        is_junk = False
        if len(d) < 200: is_junk = True
        else:
            words_500 = d.lower().split()[:500]
            unique_words = set(words_500)
            if len(unique_words) / len(words_500) < 0.2: is_junk = True
            
            alpha_count = sum(1 for c in d if c.isalpha() or c == ' ')
            if alpha_count / len(d) < 0.6: is_junk = True
            
            if any(m in d.lower() for m in ['discount code', 'coupon', 'click here to order']):
                is_junk = True
        
        if is_junk:
            score -= 3.0
            
        final_scores.append(score)

    # 3. Selection
    combined = sorted(range(len(unique_docs)), key=lambda x: final_scores[x], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    target_tokens = int(1.03 * 9830400) # We can't use T directly if it's not imported, but it is in common
    from common import T
    target_tokens = int(1.03 * T)
    
    used_in_selection = set()
    for idx in combined:
        count = doc_token_counts[idx]
        if current_tokens + count <= target_tokens:
            selected_indices.append(idx)
            current_tokens += count
            used_in_selection.add(idx)
        else:
            break
            
    # Fill to T
    if current_tokens < T:
        # Append remaining from pool (lowest scores first)
        remaining = [i for i in combined if i not in used_in_selection]
        for idx in remaining:
            selected_indices.append(idx)
            current_tokens += doc_token_counts[idx]
            if current_tokens >= T:
                break
        
        # If still short, cycle the best ones
        while current_tokens < T:
            for idx in selected_indices:
                if current_tokens >= T: break
                selected_indices.append(idx)
                current_tokens += doc_token_counts[idx]

    # 4. Order
    # Final indices are the pointers into 'docs'
    rng.shuffle(selected_indices)
    return selected_indices
