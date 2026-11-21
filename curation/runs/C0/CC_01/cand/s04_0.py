import os
import random
import numpy as np
from collections import Counter
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    enc = tiktoken.get_encoding("gpt2")
    
    def get_freq_vec(text_list):
        # Aggregate counts across all documents in a list
        counts = Counter()
        for text in text_list:
            tokens = enc.encode(text.lower())
            counts.update(tokens)
        return counts

    # Build the target vocabulary frequency vector
    target_counts = get_freq_vec(target_sample)
    
    # Prepare for cosine similarity
    # We need the unique tokens across both the target and the docs pool
    # to build a consistent feature space.
    all_indices = []
    doc_scores = []
    
    # To optimize performance, we first identify all unique tokens in the pool
    # However, since we only care about similarity to target, we only need
    # the dimensions present in the target_counts or the docs.
    # A simpler approach for cosine similarity: 
    # score(doc) = (sum(doc_counts[i] * target_counts[i])) / (norm(doc) * norm(target))
    
    target_vec = np.array([target_counts.get(i, 0) for i in range(100000)]) # Broad range
    # Better: Use only tokens that actually appear to save memory/time
    active_tokens = set(target_counts.keys())
    
    # Let's refine the approach to be more memory efficient and accurate:
    # 1. Identify all tokens in target_sample
    # 2. For each doc, count tokens
    # 3. Score = dot_product(doc_vec, target_vec) / (mag_doc * mag_target)
    
    # Optimization: Pre-calculate target vector for tokens in target
    # and only consider those tokens for the docs.
    # Actually, the dot product is only non-zero for tokens in both.
    
    # Let's collect counts for everything to be safe
    target_data = target_counts
    
    # Calculate magnitude of target vector
    target_mag = np.sqrt(sum(v**2 for v in target_data.values()))
    
    doc_scores = []
    valid_indices = []
    
    # To handle memory/speed, we calculate scores for each doc
    # We only need to iterate over tokens in target_data to calculate dot product
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc.lower())
        doc_counts = Counter(tokens)
        
        dot_product = 0.0
        doc_mag_sq = 0.0
        
        # We need to iterate over all tokens in the doc to get its magnitude
        # but only over tokens in target_data to get dot product
        for token, count in doc_counts.items():
            doc_mag_sq += count**2
            if token in target_data:
                dot_product += count * target_data[token]
        
        doc_mag = np.sqrt(doc_mag_sq)
        
        if target_mag > 0 and doc_mag > 0:
            score = dot_product / (target_mag * doc_mag)
        else:
            score = 0.0
            
        doc_scores.append(score)
        valid_indices.append(i)

    # Sort indices by score descending
    # We want high relevance first. 
    # Since we only need T tokens, and we can repeat, 
    # we pick the best docs.
    scored_indices = sorted(zip(doc_scores, valid_indices), key=lambda x: x[0], reverse=True)
    sorted_indices = [x[1] for x in scored_indices]
    
    # The trainer cycles the list, so we should provide a high-quality 
    # distribution. To ensure diversity while maintaining relevance, 
    # we could take a subset, but the prompt implies returning an 
    # ordered list. We'll return the sorted indices.
    # If the list is very long, the trainer will cycle it.
    
    return sorted_indices
