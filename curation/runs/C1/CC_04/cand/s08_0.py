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
        # Encode text and take the first 200 tokens
        tokens = enc.encode(text, mut_indices=False)[:200]
        # Create a frequency count vector for the unique tokens present in these 200
        # We use a set/dict approach to represent the "vector" of counts
        # Since we can't know the full vocab size easily for a fixed-length vector,
        # we will represent each doc as a sparse vector (dictionary) for the dot product.
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    # Build target profile
    target_counts_list = [get_token_freq_vector(doc) for doc in target_sample]
    
    # Calculate the mean "vector" (average counts per unique token across target sample)
    # To implement cosine similarity on the average, we first find all unique tokens
    # in the target sample and average their occurrences.
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
        # Norm of doc_counts
        mag_doc = np.linalg.norm(np.array(list(doc_counts.values())))
        # Norm of avg_target_vec
        mag_avg = np.linalg.norm(np.array(list(avg_target_vec.values())))
        
        similarity = dot_product / (mag_doc * mag_avg + 1e-9)
        scores.append((similarity, i))
    
    # Sort by similarity descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top half to ensure diversity and volume
    top_indices = [idx for score, idx in scores[:len(docs)//2]]
    
    # Shuffle to avoid ordering bias while keeping high quality
    rng.shuffle(top_indices)
    
    return top_indices
