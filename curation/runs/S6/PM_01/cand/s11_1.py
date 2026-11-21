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

    # Pre-calculate token counts to avoid re-encoding
    doc_token_counts = [count_tokens(d) for d in docs]
    
    # Basic Filtering: length and structural junk
    # We use a faster check than regex for most documents
    def is_junk(text):
        text_strip = text.strip()
        if len(text_strip) < 50: return True
        # Fast check for common tags
        if '[' in text_strip and ']' in text_strip: return True
        if '{' in text_strip and '}' in text_strip: return True
        if '<' in text_strip and '>' in text_strip: return True
        return False

    # Use a smaller, faster model for scoring
    # We limit the number of docs processed to save time
    # and use a simpler feature set.
    sample_size = min(len(docs), 500)
    sample_indices = rng.sample(range(len(docs)), sample_size)
    
    # Simple Unigram vectorizer for speed
    vectorizer = CountVectorizer(analyzer='word', max_features=2000)
    
    # Train on a mix of target and a few random docs
    train_texts = target_sample + [docs[i] for i in sample_indices]
    y = [1] * len(target_sample) + [0] * (len(train_texts) - len(target_sample))
    
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(solver='liblinear', max_iter=500)
    clf.fit(X_train, y)
    
    X_pool = vectorizer.transform(docs)
    scores = clf.predict_proba(X_pool)[:, 1]

    # Filtering: Keep docs that are high score AND not junk
    candidate_indices = []
    for i in range(len(docs)):
        if not is_junk(docs[i]) and scores[i] > 0.2:
            candidate_indices.append(i)
    
    # Sort by score descending
    candidate_indices.sort(key=lambda x: scores[x], reverse=True)
    
    # Selection with token budget
    selected_indices = []
    current_tokens = 0
    
    # Pass 1: Fill as much as possible with highest quality
    for idx in candidate_indices:
        count = doc_token_counts[idx]
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            
    # Cycling logic: if we have candidates, cycle them to reach T
    if selected_indices and current_tokens < T:
        # Use a loop to cycle the best candidates found
        while current_tokens < T and len(selected_indices) > 0:
            for idx in selected_indices:
                count = doc_token_counts[idx]
                if current_tokens + count <= T:
                    selected_indices.append(idx)
                    current_tokens += count
                if current_tokens >= T:
                    break
            if current_tokens >= T:
                break
                
    # If still under T, fill with any non-junk docs (fallback)
    if current_tokens < T:
        for i in range(len(docs)):
            if i not in selected_indices and not is_junk(docs[i]):
                count = doc_token_counts[i]
                if current_tokens + count <= T:
                    selected_indices.append(i)
                    current_tokens += count
                if current_tokens >= T:
                    break

    # Final shuffle to avoid ordering bias
    rng.shuffle(selected_indices)
    return selected_indices
