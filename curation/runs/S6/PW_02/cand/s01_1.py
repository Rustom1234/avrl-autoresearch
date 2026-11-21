import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Simple tokenization
    def get_tokens(text):
        return text.lower().split()

    # We need a fast way to score documents to find the best content.
    # Tfidf + Logistic Regression is too slow/memory heavy for 2T pool.
    # We use a simple keyword frequency approach based on the target sample.
    
    target_tokens = []
    for d in target_sample:
        target_tokens.extend(get_tokens(d))
    
    # Create a frequency map for target domain words
    word_counts = {}
    for token in target_tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
        
    # Filter out very common English words to focus on domain-specific signals
    stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "to", "of", "with", "at", "by", "this", "that", "from", "it", "in", "on", "for"}
    for sw in stop_words:
        if sw in word_counts:
            del word_counts[sw]

    # Score the pool
    # We want to find docs that have high overlap with target_sample words
    pool_scores = []
    valid_indices = []
    
    # Pre-calculate some constants
    # We'll cap the pool size to keep it within 180 seconds
    # 2T tokens is roughly 2 * 9.8M = 19.6M tokens.
    # We want to find enough documents to fill T tokens.
    
    for i, d in enumerate(docs):
        tokens = get_tokens(d)
        if len(tokens) < 20:
            continue
            
        score = 0
        count = 0
        for token in tokens:
            if token in word_counts:
                score += word_counts[token]
                count += 1
        
        # Normalize score by length to avoid favoring very long junk docs
        # but also ensure we aren't penalizing high-quality long docs too much.
        norm_score = score / (len(tokens) ** 0.5)
        pool_scores.append((norm_score, i))
        valid_indices.append(i)

    # Sort by score descending
    pool_scores.sort(key=lambda x: x[0], reverse=True)
    
    # We need to select enough docs to reach T tokens.
    # Since we don't know exact token count of docs easily without processing,
    # and T is large, we take a large enough sample.
    # The prompt says training stops after T tokens, and we can cycle.
    # To maximize quality, we take the best docs until we have a healthy amount.
    
    selected_indices = []
    total_tokens_estimate = 0
    # We want to select indices until we're likely to exceed T. 
    # Let's take a safe number of high-quality docs.
    # If we take 1000 documents, it's likely to be a good variety.
    # But let's try to get a lot of variety.
    
    limit = min(len(pool_scores), 5000) 
    for i in range(limit):
        selected_indices.append(pool_scores[i][1])
    
    # If we have very few, take more
    if len(selected_indices) < 1000 and len(pool_scores) > 1000:
        selected_indices.extend([pool_scores[i][1] for i in range(1000, min(len(pool_scores), 2000))])

    # Shuffle a portion to ensure variety, but keep top quality
    rng.shuffle(selected_indices)
    
    # Ensure we return a list
    return selected_indices
