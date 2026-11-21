import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken for gpt2 encoding
    enc = tiktoken.get_encoding("gpt2")
    
    def get_token_freq_vector(text):
        # Removed mut_indices=False as it is not a valid argument for the encoding object
        tokens = enc.encode(text)[:200]
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    # Build target profile
    target_counts_list = [get_token_freq_vector(doc) for doc in target_sample]
    
    # Calculate the mean "vector"
    all_target_tokens = set()
    for counts in target_counts_list:
        all_target_tokens.update(counts.keys())
    
    avg_target_vec = {}
    num_targets = len(target_counts_list)
    if num_targets == 0:
        return list(range(len(docs)))
        
    for t in all_target_tokens:
        avg_target_vec[t] = sum(counts.get(t, 0) for counts in target_counts_list) / num_targets

    # Score documents in the pool
    scores = []
    for i, doc in enumerate(docs):
        doc_counts = get_token_freq_vector(doc)
        
        # Dot product of doc_counts and avg_target_vec
        dot_product = 0.0
        for t, count in doc_counts.items():
            if t in avg_target_vec:
                dot_product += count * avg_target_vec[t]
        
        # Calculate norms
        doc_vals = list(doc_counts.values())
        mag_doc = np.linalg.norm(doc_vals) if doc_vals else 0.0
        
        avg_vals = list(avg_target_vec.values())
        mag_avg = np.linalg.norm(avg_vals) if avg_vals else 0.0
        
        similarity = dot_product / (mag_doc * mag_avg + 1e-9)
        scores.append((similarity, i))
    
    # Sort by similarity descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top half to ensure diversity and volume
    num_to_take = len(docs) // 2
    if num_to_take == 0:
        num_to_take = 1
    top_indices = [idx for score, idx in scores[:num_to_take]]
    
    # Shuffle to avoid ordering bias while keeping high quality
    rng.shuffle(top_indices)
    
    return top_indices
