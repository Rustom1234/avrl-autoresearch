import os
import random
import zlib
import re
import hashlib
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Normalisation and Initial Filtering
    # Keep a dictionary to handle exact/harary duplicate removal via hash
    unique_docs = {}
    for i, doc in enumerate(docs):
        norm = doc.lower().strip()
        # Simple whitespace normalization for duplicate check
        norm_ws = " ".join(norm.split())
        h = hashlib.md5(norm_ws.encode()).hexdigest()
        if h not in unique_docs:
            unique_docs[h] = (i, norm)
            
    # Filter: shorter than 200 chars
    filtered_indices = []
    for h, (original_idx, norm_text) in unique_docs.items():
        if len(norm_text) >= 200:
            filtered_indices.append((original_idx, norm_text))

    def get_tokens(text):
        return text.lower().split()

    def get_bigrams(tokens):
        return set(zip(tokens, tokens[1:]))

    # 2a & 3. Bigram Score and Calibration
    # Build bigrams from target_sample
    target_bigrams = set()
    target_texts_lower = []
    for ts in target_sample:
        tokens = get_tokens(ts)
        target_texts_lower.append(ts.lower())
        target_bigrams.update(get_bigrams(tokens))

    def calculate_bigram_score(text):
        tokens = get_tokens(text)
        if len(tokens) < 2: return 0.0
        # Limit to first 3000 words
        subset = tokens[:3000]
        bi_count = 0
        bi_total = 0
        for i in range(len(subset)-1):
            bi = (subset[i], subset[i+1])
            if bi in target_bigrams:
                bi_count += 1
            bi_total += 1
        return bi_count / bi_total if bi_total > 0 else 0.0

    # Score target_sample for calibration
    target_scores = [calculate_bigram_score(ts.lower()) for ts in target_sample]
    mu = np.mean(target_scores)
    sigma = np.std(target_scores)

    # Score pool docs
    scored_docs = []
    for idx, text in filtered_indices:
        # 2b. Compression detour
        # Normalize punctuation but keep whitespace
        punc_stripped = re.sub(r'[^\w\s]', '', text.lower())
        compressed = zlib.compress(punc_stripped.encode('utf-8'))
        comp_ratio = len(compressed) / len(punc_stripped) if len(punc_stripped) > 0 else 1.0
        
        if comp_ratio > 0.25:
            continue
            
        b_score = calculate_bigram_score(text)
        scored_docs.append({
            'idx': idx,
            'text': text,
            'b_score': b_score,
            'comp_ratio': comp_ratio
        })

    # 3. Scoring and Selection logic
    # Character 3-grams for HashingVectorizer
    def get_3grams(text):
        return [text[i:i+3].lower() for i in range(len(text)-2)]

    # HashingVectorizer for similarity
    hv = HashingVectorizer(n_gram_range=(3, 3))
    hv.fit(target_sample)
    target_vecs = hv.transform(target_sample).toarray()
    
    # Leave-half-out logic (simplified as mean of half)
    # We'll treat the first half of target_sample as "held-out" for similarity scoring
    half_idx = len(target_sample) // 2
    hv.fit(target_sample[:half_idx] + target_sample[half_idx:])
    
    pool_scored = []
    for item in scored_docs:
        # Similarity to full target
        vec = hv.transform([item['text']]).toarray()
        sim = np.mean(np.dot(vec, target_vecs.T))
        
        # Score for ranking
        pool_scored.append({
            'idx': item['idx'],
            'text': item['text'],
            'b_score': item['b_score'],
            'sim': sim,
            'tokens': len(item['text']) // 4
        })

    # Sorting criteria: 1. b_score >= mu+0.5sigma, 2. b_score >= mu+0.2sigma, 3. Sim
    high_quality = [d for d in pool_scored if d['b_score'] >= (mu + 0.5 * sigma)]
    medium_quality = [d for d in pool_scored if d['b_score'] >= (mu + 0.2 * sigma) and d['b_score'] < (mu + 0.5 * sigma)]
    remaining = [d for d in pool_scored if d['b_score'] < (mu + 0.2 * sigma)]
    
    # Sort remaining by similarity
    remaining.sort(key=lambda x: x['sim'], reverse=True)
    
    # Combine
    combined = high_quality + medium_quality + remaining
    # Sort all by b_score then similarity
    combined.sort(key=lambda x: (x['b_score'], x['sim']), reverse=True)

    # 4. Token counting and final pruning
    total_tokens = 0
    selected_indices = []
    
    # First pass: Take high quality stuff
    # Actually, the instructions say: "Take the top-scoring docs... 1.0T tokens at most"
    # and "If the docs above the mu+0.5sigma threshold hold fewer than 1T tokens, 
    # keep all of them... If they hold more than 1T... shoot down"
    
    # Re-evaluating logic:
    # High quality = b_score >= mu + 0.5*sigma
    hq_docs = [d for d in pool_scored if d['b_score'] >= (mu + 0.5 * sigma)]
    hq_tokens = sum(d['tokens'] for d in hq_docs)
    
    if hq_tokens < 1000000: # 1T / 10? Wait, T = 9,830,400, so 1T is 10^12? 
        # User says 1.0T tokens. In the context of T=9,830,400, "1.0T" likely means 1.0 * T.
        # Let's assume 1.0T = 1.0 * T = 9,830,400.
        pass
    
    # Re-reading: "Take the top-scoring docs, about 1.0T tokens at most."
    # Let's interpret 1.0T as 1.0 * T = 9,830,400.
    limit = T 
    
    # If hq_docs < limit tokens, return hq_docs shuffled
    if hq_tokens < limit:
        # "keep all of them, shuffle with seeded random"
        # Note: the prompt says "If the docs above the mu+0.5sigma threshold hold fewer than 1T tokens, 
        # keep all of them... letting the trainer cycle"
        # This implies we DON'T add the remaining docs in this specific branch.
        selected_indices = [d['idx'] for d in hq_docs]
    else:
        # "If they hold more than 1T tokens, rank once more by a second leave-half-out 
        # target classifier... and shoot down the top-scoring docs until cumulative token count reaches 1T."
        # Logistic Regression on char 2-4 grams
        pos_docs = target_sample[:len(target_sample)//2]
        neg_docs = docs[0:min(len(docs), 20000)]
        
        # Helper to get 2-4 grams
        def get_ngrams(text, n):
            return [text[i:i+n].lower() for i in range(len(text)-n+1)]
        
        X_train = []
        y_train = []
        for d in pos_docs:
            X_train.append(" ".join(get_ngrams(d, 2) + get_ngrams(d, 3) + get_ngrams(d, 4)))
            y_train.append(1)
        for d in neg_docs:
            X_train.append(" ".join(get_ngrams(d, 2) + get_ngrams(d, 3) + get_ngrams(d, 4)))
            y_train.append(0)
            
        lr = LogisticRegression(max_iter=500)
        hv_lr = HashingVectorizer(n_gram_range=(2, 4))
        X_train_vec = hv_lr.transform(X_train).toarray()
        lr.fit(X_train_vec, y_train)
        
        # Score all pool docs
        pool_lr_scores = []
        for d in pool_scored:
            vec = hv_lr.transform([d['text']]).toarray()
            score = lr.predict_proba(vec)[0][1]
            # Scale uniformly to [0,1] - since predict_proba is already 0-1, we just use it
            pool_lr_scores.append((d['idx'], score, d['tokens']))
            
        # Sort by lr score
        pool_lr_scores.sort(key=lambda x: x[1], reverse=True)
        
        current_tokens = 0
        for idx, score, tokens in pool_lr_scores:
            if current_tokens + tokens <= limit:
                selected_indices.append(idx)
                current_tokens += tokens
            else:
                break
                
    rng.shuffle(selected_indices)
    return selected_indices
