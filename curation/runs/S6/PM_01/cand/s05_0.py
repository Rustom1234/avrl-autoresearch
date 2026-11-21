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

    # 1. Structural Filter (WikiText features)
    # High-quality text usually has a balance of punctuation, sentence length, and non-junk chars.
    def is_not_junk(text):
        # Basic heuristic: length, presence of common punctuation, and alpha ratio
        if len(text) < 50: return False
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / len(text) < 0.7: return False
        return True

    # 2. Prepare Classifier
    # Chunking target_sample to ~1000 examples (max_features=10000 for sparsity/speed)
    pos_texts = []
    chunk_size = max(1, len(target_sample) // 1000)
    for i in range(0, len(target_sample), chunk_size):
        chunk = "".join(target_sample[i:i+chunk_size])
        pos_texts.append(chunk[:1000])

    # Sample some random docs from the pool as negatives
    # We want a representative set for the classifier
    neg_count = min(len(docs), 1000)
    neg_indices = rng.sample(range(len(docs)), neg_count)
    neg_texts = [docs[i][:1000] for i in neg_indices]

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
    X = vectorizer.fit_transform(pos_texts + neg_texts)
    y = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    # Train with C for regularization
    clf = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    clf.fit(X, y)

    # 3. Score and Filter
    scores = []
    valid_indices = []
    
    for i, d in enumerate(docs):
        if is_not_junk(d):
            score = clf.predict_proba(vectorizer.transform([d[:1000]]))[0, 1]
            scores.append(score)
            valid_indices.append(i)
        else:
            scores.append(-1.0)
            valid_indices.append(i)

    # Sort indices by classifier score
    ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)

    # 4. Selection Logic
    selected_indices = []
    current_tokens = 0
    
    # Get indices that pass both structural and similarity checks
    high_quality_indices = [ranked_indices[i] for i in range(len(ranked_indices)) if scores[ranked_indices[i]] > 0.1]
    
    # First pass: Fill T with high-quality documents
    for idx in high_quality_indices:
        cnt = count_tokens(docs[idx])
        if current_tokens + cnt <= T:
            selected_indices.append(idx)
            current_tokens += cnt
        else:
            break

    # If we still need more, pad with the next best ones from ranked_indices
    if current_tokens < T:
        # To ensure we don't include junk, we skip indices that failed is_not_junk
        for idx in ranked_indices:
            if idx in selected_indices: continue
            if not is_not_junk(docs[idx]): continue
            
            cnt = count_tokens(docs[idx])
            if current_tokens + cnt <= T:
                selected_indices.append(idx)
                current_tokens += cnt
            else:
                break

    # Final check: if still under T (rare with cycle), take from best and shuffle
    if current_tokens < T:
        best_pool = [idx for idx in high_quality_indices if idx not in selected_indices]
        if not best_pool: best_pool = ranked_indices[:50]
        
        rng.shuffle(best_pool)
        for idx in best_pool:
            cnt = count_tokens(docs[idx])
            if current_tokens + cnt <= T:
                selected_indices.append(idx)
                current_tokens += cnt
            else:
                break

    # Shuffle the final selection to mix the order
    rng.shuffle(selected_indices)
    return selected_indices
