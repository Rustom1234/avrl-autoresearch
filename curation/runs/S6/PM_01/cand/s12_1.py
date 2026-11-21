import os
import random
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    def get_features(text):
        words = text.split()
        if not words: return np.array([0.0, 0.0, 0.0])
        lengths = [len(w) for w in words]
        return np.array([
            float(len(words)),
            float(sum(lengths) / len(lengths)),
            float(np.std(lengths) if len(lengths) > 1 else 0.0)
        ])

    # Extract features for the target sample
    pos_feats = np.array([get_features(d) for d in target_sample])
    
    # Calculate centroid of target sample
    pos_centroid = pos_feats.mean(axis=0)
    
    # Evaluate all docs in the pool
    # We prioritize docs that are structurally similar to the target sample
    # and have a reasonable length (not too short, not too long)
    doc_scores = []
    doc_token_counts = []
    
    for i, doc in enumerate(docs):
        feat = get_features(doc)
        # Distance to target centroid
        dist = np.linalg.norm(feat - pos_centroid)
        
        # Heuristic: high word count and moderate length/variance is often good quality
        # We weigh distance to centroid heavily
        # Score = 1 / (1 + distance)
        score = 1.0 / (1.0 + dist)
        
        # Penalty for very short or very long sequences (potential junk)
        word_count = len(doc.split())
        if word_count < 20 or word_count > 1000:
            score *= 0.1
            
        doc_scores.append(score)
        doc_token_counts.append(count_tokens(doc))

    # Combine scores with a slight preference for content richness
    # (Log of word count to prevent extreme outliers)
    final_scores = np.array(doc_scores)
    indices = np.argsort(final_scores)[::-1]
    
    selected_indices = []
    current_total_tokens = 0
    
    # Greedily select top docs until T is reached
    # We use a two-pass approach to ensure we fill the budget efficiently
    # Pass 1: High quality docs
    for idx in indices:
        count = doc_token_counts[idx]
        if current_total_tokens + count <= T:
            selected_indices.append(idx)
            current_total_tokens += count
            
    # Pass 2: Fill remaining space with remaining docs
    remaining_indices = [i for i in indices if i not in selected_indices]
    for idx in remaining_indices:
        count = doc_token_counts[idx]
        if current_total_tokens + count <= T:
            selected_indices.append(idx)
            current_total_tokens += count
        else:
            # If a doc is too big to fit perfectly, we don't skip it if it's 
            # the very next best, but the prompt says "repeats allowed" 
            # and "stops after T", so we can just stop or cycle.
            pass
            
    # If we still haven't reached T (due to large individual docs), 
    # cycle the best ones we found.
    if current_total_tokens < T and selected_indices:
        while current_total_tokens < T and selected_indices:
            for idx in selected_indices:
                count = doc_token_counts[idx]
                if current_total_tokens + count <= T:
                    selected_indices.append(idx)
                    current_total_tokens += count
                if current_total_tokens >= T:
                    break
            if current_total_tokens >= T:
                break
                
    # Shuffle to ensure variety in the training sequence
    random.shuffle(selected_indices)
    return selected_indices
