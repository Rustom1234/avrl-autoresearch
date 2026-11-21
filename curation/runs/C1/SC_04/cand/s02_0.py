import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    
    # Pre-filter step (Step 4)
    filtered_indices = []
    seen_keys = set()
    
    for i, d in enumerate(docs):
        # Length check
        if len(d) < 200:
            continue
        
        # Character sanity check (85% ASCII letters, digits, or whitespace)
        count = 0
        for char in d:
            if char.isalnum() or char.isspace():
                count += 1
        if count / len(d) < 0.85:
            continue
            
        # Diversity check (unique words / total words >= 0.25 on first 3000 chars)
        chunk = d[:3000]
        words = chunk.split()
        if len(words) > 0:
            unique_words = len(set(w for w in words if w))
            if unique_words / len(words) < 0.25:
                continue
        else:
            continue
            
        # Deduplication
        key = ' '.join(d.lower().split())[:200]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        filtered_indices.append(i)

    # Vectorizer and Classifier (Steps 1, 2, 3)
    clip = lambda s: s[:2000]
    pos_texts = [clip(d) for d in target_sample]
    
    neg_count = min(3000, len(docs))
    neg_indices = rng.sample(range(len(docs)), neg_count)
    neg_texts = [clip(docs[i]) for i in neg_indices]
    
    all_texts = pos_texts + neg_texts
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), 
                           sublinear_tf=True, max_features=50000, lowercase=False)
    Xtr = vec.fit_transform(all_texts)
    y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
    assert Xtr.shape[0] == len(y)
    
    clf = LogisticRegression(C=10, class_weight='balanced', max_iter=1000)
    clf.fit(Xtr, y)
    
    # Scoring (Step 3 & 5)
    enc = tiktoken.get_encoding("gpt2")
    scores = []
    valid_indices = []
    
    # Process docs in chunks of 2000 to stay efficient
    for i in range(0, len(docs), 2000):
        chunk_indices = filtered_indices[i:i+2000]
        chunk_texts = [docs[idx] for idx in chunk_indices]
        # Need to filter the indices again to only those that passed step 4
        # But our filtered_indices already only contains those that passed step 4.
        
        # Since we need scores for the survivors, let's re-evaluate only survivors
        pass

    # Refined logic: Only score the survivors
    survivor_docs = [docs[i] for i in filtered_indices]
    # Score survivors
    scores = []
    for d in survivor_docs:
        scores.append(clf.decision_function(vec.transform([clip(d)]))[0])
    
    # Tokens
    token_counts = enc.encode_ordinary_batch([d for d in survivor_docs])
    doc_tokens = [len(tc) for tc in token_counts]
    
    # Combine data
    data = []
    for i in range(len(filtered_indices)):
        data.append({
            'idx': filtered_indices[i],
            'score': scores[i],
            'tokens': doc_tokens[i]
        })
    
    # Sort by score descending
    data.sort(key=lambda x: x['score'], reverse=True)
    
    # Selection (Step 6)
    selected_indices = []
    current_tokens = 0
    
    # First pass: scores > 0
    for item in data:
        if item['score'] > 0:
            selected_indices.append(item['idx'])
            current_tokens += item['tokens']
        else:
            break
            
    # Second pass: fill to T
    if current_tokens < T:
        for item in data:
            if item['score'] <= 0:
                selected_indices.append(item['idx'])
                current_tokens += item['tokens']
            if current_tokens >= T:
                break
                
    # Final shuffle
    final_indices = selected_indices
    rng.shuffle(final_indices)
    return final_indices
