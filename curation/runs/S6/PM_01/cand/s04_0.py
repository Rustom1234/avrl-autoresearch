import os
import random
import re
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Encoder for token counting
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    # Prepare Training Data for LR
    # Positives: chunked 2000s
    pos_texts = []
    for i in range(0, len(target_sample), 2000):
        chunk = "".join(target_sample[i:i+2000])
        pos_texts.append(chunk[:4000])
    
    # Negatives: 1500 random docs
    neg_count = min(1500, len(docs))
    neg_indices = rng.sample(list(range(len(docs))), neg_count)
    neg_texts = [docs[i][:4000] for i in neg_indices]
    
    # Vectorizer: char n-grams (1-3)
    # We use a smaller max_features to stay within CPU/time limits
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=10000)
    
    # Train LR
    X = vectorizer.fit_transform(pos_texts + neg_texts)
    y = [1] * len(pos_texts) + [0] * len(neg_texts)
    clf = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf.fit(X, y)
    
    # Score all pool docs
    pool_texts = [d[:4000] for d in docs]
    X_pool = vectorizer.transform(pool_texts)
    scores = clf.predict_proba(X_pool)[:, 1]
    
    # Rank indices
    ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    
    # Calculate selection
    # Goal: keep adding top docs until we hit T tokens.
    # Logic: cycle if under 0.4T, no padding if less (meaning we take exactly enough)
    # Since we want to maximize quality, we take the best until T is reached.
    
    selected_indices = []
    current_tokens = 0
    limit = T
    
    # We use the ranked list. To handle the "cycling if under 0.4T" logic:
    # If we reach 0.4T and haven't filled, we cycle the start.
    # However, the prompt implies ranking to fill T.
    
    # Greedily pick from ranked indices
    for idx in ranked_indices:
        count = count_tokens(docs[idx])
        if current_tokens + count <= limit:
            selected_indices.append(idx)
            current_tokens += count
        else:
            # If we can't fit the whole doc, we don't add it (standard behavior)
            # But to ensure we hit T, we check if we need to cycle
            pass

    # Ensure we reach T tokens. If we are under 0.4T, cycle.
    # This is best handled by creating a sequence that fills T.
    final_indices = []
    current_total = 0
    # Re-evaluating: "take top docs until hitting T tokens (cycling if under 0.4T, no padding if less)"
    # This suggests a specific construction:
    
    # 1. Take top docs until T or until we run out.
    # 2. If total < 0.4T, cycle the indices.
    
    # Let's build a flat list of indices that covers T
    result = []
    temp_total = 0
    for idx in ranked_indices:
        cnt = count_tokens(docs[idx])
        result.append(idx)
        temp_total += cnt
        if temp_total >= limit:
            # Trim the last one if it exceeds T to be precise, 
            # though "no padding" usually means we don't pad with empty space.
            break
    
    # Cycle logic
    if temp_total < 0.4 * T:
        # Cycle the sequence until it's long enough or we have exhausted docs
        while temp_total < T:
            for idx in result:
                cnt = count_tokens(docs[idx])
                result.append(idx)
                temp_total += cnt
                if temp_total >= T:
                    break
            if not result: break # safety
            
    # Final trim to be as close to T as possible without exceeding significantly
    # or just keep it as is if it's within reasonable bounds.
    # The prompt says "stops after T tokens", so the list length doesn't matter as much as content.
    
    # Shuffle final list
    rng.shuffle(result)
    return result
