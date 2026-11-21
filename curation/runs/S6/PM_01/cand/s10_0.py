import os
import random
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    def normalize(text):
        return re.sub(r'\s+', ' ', text.lower()).strip()

    def get_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        return set(zip(words, words[1:]))

    # Pre-calculate target bigrams
    target_bigrams = set()
    for s in target_sample:
        target_bigrams.update(get_bigrams(s))

    # 1. Strict Deduplication and Junk Removal
    clean_indices = []
    seen_hashes = set()
    
    # We iterate through docs to find high-quality candidates
    for i, d in enumerate(docs):
        norm_d = normalize(d)
        # Junk check: min 200 chars
        if len(norm_d) < 200:
            continue
        
        # Deduplication
        h = hash(norm_d)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Bigram hit rate >= 0.3
        bgs = get_bigrams(norm_d)
        denom = min(len(bgs), 199)
        hit_rate = (len(bgs.intersection(target_bigrams)) / denom) if denom > 0 else 0.0
        
        if hit_rate >= 0.3:
            clean_indices.append(i)

    if not clean_indices:
        # Fallback to raw docs if cleaning is too strict
        clean_indices = list(range(len(docs)))

    # 2. Scoring with Logistic Regression
    # Use a sample of target_sample and cleaned docs to train
    train_texts = target_sample + [docs[i] for i in clean_indices[:min(len(clean_indices), 400)]]
    y = [1] * len(target_sample) + [0] * min(len(clean_indices), 400)
    
    # Fast vectorization
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_train, y)
    
    # Score all docs
    X_pool = vectorizer.transform(docs)
    scores = clf.predict_proba(X_pool)[:, 1]
    
    # Rank indices by score
    ranked_indices = sorted(clean_indices, key=lambda i: scores[i], reverse=True)
    
    # 3. Selection
    selected_indices = []
    current_tokens = 0
    
    # Primary selection
    for idx in ranked_indices:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # 4. Limited padding if under 0.5T
    if current_tokens < 0.5 * T:
        # Find docs in ranked_indices not already selected
        selected_set = set(selected_indices)
        for idx in ranked_indices:
            if idx not in selected_set:
                count = count_tokens(docs[idx])
                if current_tokens + count <= T:
                    selected_indices.append(idx)
                    current_tokens += count
                else:
                    break
                    
    # Cycle logic to fill exactly T tokens
    if current_tokens < T:
        to_cycle = selected_indices if selected_indices else ranked_indices[:10]
        while current_tokens < T and to_cycle:
            for idx in to_cycle:
                count = count_tokens(docs[idx])
                if current_tokens + count <= T:
                    selected_indices.append(idx)
                    current_tokens += count
                if current_tokens >= T:
                    break
                    
    rng.shuffle(selected_indices)
    return selected_indices
