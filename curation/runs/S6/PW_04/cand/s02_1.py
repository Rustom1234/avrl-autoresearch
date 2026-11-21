import os
import random
import zlib
import re
import hashlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Normalisation and Initial Filtering
    unique_docs = {}
    for i, doc in enumerate(docs):
        norm = doc.lower().strip()
        norm_ws = " ".join(norm.split())
        h = hashlib.md5(norm_ws.encode()).hexdigest()
        if h not in unique_docs:
            unique_docs[h] = (i, norm)
            
    filtered_indices = []
    for h, (original_idx, norm_text) in unique_docs.items():
        if len(norm_text) >= 200:
            filtered_indices.append((original_idx, norm_text))

    def get_tokens(text):
        return text.lower().split()

    def get_bigrams(tokens):
        return set(zip(tokens, tokens[1:]))

    # 2a & 3. Bigram Score and Calibration
    target_bigrams = set()
    for ts in target_sample:
        tokens = get_tokens(ts)
        target_bigrams.update(get_bigrams(tokens))

    def calculate_bigram_score(text):
        tokens = get_tokens(text)
        if len(tokens) < 2: return 0.0
        subset = tokens[:3000]
        bi_count = 0
        bi_total = 0
        for i in range(len(subset)-1):
            bi = (subset[i], subset[i+1])
            if bi in target_bigrams:
                bi_count += 1
            bi_total += 1
        return bi_count / bi_total if bi_total > 0 else 0.0

    target_scores = [calculate_bigram_score(ts.lower()) for ts in target_sample]
    mu = np.mean(target_scores)
    sigma = np.std(target_scores)

    scored_docs = []
    for idx, text in filtered_indices:
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
            'tokens': len(text.split())
        })

    # Using HashingVectorizer without n_gram_range for similarity
    hv = HashingVectorizer()
    target_vecs = hv.transform(target_sample).toarray()
    
    pool_scored = []
    for item in scored_docs:
        vec = hv.transform([item['text']]).toarray()
        sim = np.mean(np.dot(vec, target_vecs.T))
        
        pool_scored.append({
            'idx': item['idx'],
            'text': item['text'],
            'b_score': item['b_score'],
            'sim': sim,
            'tokens': item['tokens']
        })

    hq_docs = [d for d in pool_scored if d['b_score'] >= (mu + 0.5 * sigma)]
    hq_tokens = sum(d['tokens'] for d in hq_docs)
    
    limit = T 
    
    if hq_tokens < limit:
        selected_indices = [d['idx'] for d in hq_docs]
    else:
        # Logistic Regression logic
        pos_docs = target_sample[:len(target_sample)//2]
        # Take a subset of docs for the negative class
        neg_docs = docs[0:min(len(docs), 5000)]
        
        def get_ngrams_manual(text, n):
            return [text[i:i+n].lower() for i in range(len(text)-n+1)]
        
        X_train = []
        y_train = []
        for d in pos_docs:
            X_train.append(" ".join(get_ngrams_manual(d, 2) + get_ngrams_manual(d, 3)))
            y_train.append(1)
        for d in neg_docs:
            X_train.append(" ".join(get_ngrams_manual(d, 2) + get_ngrams_manual(d, 3)))
            y_train.append(0)
            
        hv_lr = HashingVectorizer()
        X_train_vec = hv_lr.transform(X_train).toarray()
        lr = LogisticRegression(max_iter=500, solver='liblinear')
        lr.fit(X_train_vec, y_train)
        
        pool_lr_scores = []
        for d in pool_scored:
            vec = hv_lr.transform([d['text']]).toarray()
            score = lr.predict_proba(vec)[0][1]
            pool_lr_scores.append((d['idx'], score, d['tokens']))
            
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
