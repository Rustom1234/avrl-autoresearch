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

    # Tokenization and Feature Extraction
    # We use word bigrams as requested
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
    
    # Prepare data for Logistic Regression
    # Train on target_sample + a smaller random portion of the pool to help with boundary
    neg_pool_size = min(len(docs), 500)
    neg_indices = rng.sample(list(range(len(docs))), neg_pool_size)
    
    train_texts = target_sample + [docs[i] for i in neg_indices]
    y = [1] * len(target_sample) + [0] * neg_pool_size
    
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y)
    
    # Score all docs
    X_pool = vectorizer.transform(docs)
    scores = clf.predict_proba(X_pool)[:, 1]
    
    # Bigram Hit Rate logic
    # Get bigrams from target_sample
    def get_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        return set(zip(words, words[1:]))

    target_bigrams = set()
    for s in target_sample:
        target_bigrams.update(get_bigrams(s))
        
    def get_first_200_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        window = words[:200]
        return set(zip(window, window[1:]))

    # Calculate hit rates for all docs
    doc_hit_rates = []
    for d in docs:
        bgs = get_first_200_bigrams(d)
        hits = len(bgs.intersection(target_bigrams))
        # Normalize by number of bigrams in the window (max 199)
        denom = min(len(bgs), 199)
        doc_hit_rates.append(hits / denom if denom > 0 else 0.0)

    # Ranking and Filtering
    # Rank indices by LR scores
    ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    
    # Find 25th percentile of hit rates among top-scoring docs
    # We take the top 50% of scores to find the "good" ones
    top_count = max(1, len(docs) // 2)
    top_indices = ranked_indices[:top_count]
    top_hit_rates = [doc_hit_rates[i] for i in top_indices]
    threshold = np.percentile(top_hit_rates, 25)
    
    # Filter docs
    valid_indices = []
    for i in range(len(docs)):
        if doc_hit_rates[i] >= threshold:
            valid_indices.append(i)
            
    # Re-rank only valid indices by score
    valid_ranked = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
    
    # Select indices to reach T tokens
    selected_indices = []
    current_tokens = 0
    
    for idx in valid_ranked:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # Cycle logic
    if current_tokens < T:
        to_cycle = selected_indices if selected_indices else valid_ranked[:10]
        while current_tokens < T and to_cycle:
            for idx in to_cycle:
                count = count_tokens(docs[idx])
                selected_indices.append(idx)
                current_tokens += count
                if current_tokens >= T:
                    break
                    
    rng.shuffle(selected_indices)
    return selected_indices
