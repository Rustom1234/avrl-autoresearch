import os
import random
import collections
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_ngram_vector(text):
        # Use first 300 characters to build a bigram count vector
        chunk = text[:300].lower()
        # Use a set of bigrams to build a vocabulary for this run
        # To ensure consistency across target and pool, we must derive 
        # the vocabulary from the union of the sample and the docs.
        # However, since we can't easily pre-scan everything, we'll 
        # use a fixed set of common bigrams or a combined approach.
        # To strictly follow "bigram count vector" logic:
        bigrams = []
        for i in range(len(chunk) - 1):
            bigrams.append(chunk[i:i+2])
        
        # We'll use a dynamic vocabulary based on the target_sample 
        # to build the centroid, then map the pool to that same space.
        return bigrams

    # Extract bigrams from target sample
    target_bigrams_list = []
    for doc in target_sample:
        chunk = doc[:300].lower()
        bg = [chunk[i:i+2] for i in range(len(chunk)-1)]
        target_bigrams_list.append(bg)

    # Create a global vocabulary from the target sample
    vocab = sorted(list(set([bg for bg_list in target_bigrams_list for bg in bg_list])))
    vocab_map = {bg: i for i, bg in enumerate(vocab)}
    vocab_size = len(vocab)

    def get_vector_fixed(bg_list):
        vec = np.zeros(vocab_size)
        for bg in bg_list:
            if bg in vocab_map:
                vec[vocab_map[bg]] += 1
        return vec

    # Build target centroid
    target_vectors = [get_vector_fixed(bg_list) for bg_list in target_bigrams_list]
    if not target_vectors:
        return list(range(len(docs)))
    
    avg_vec = np.mean(target_vectors, axis=0)
    
    # Score pool documents
    scored_indices = []
    for i, doc in enumerate(docs):
        chunk = doc[:300].lower()
        bg_list = [chunk[j:j+2] for j in range(len(chunk)-1)]
        v = get_vector_fixed(bg_list)
        
        dot = np.dot(v, avg_vec)
        mag_v = np.linalg.norm(v)
        mag_avg = np.linalg.norm(avg_vec)
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select top indices
    # Taking a large enough portion to ensure we have plenty of tokens
    # while prioritizing high similarity.
    top_count = min(len(docs), len(docs) // 2)
    top_indices = [idx for score, idx in scored_indices[:top_count]]
    
    rng.shuffle(top_indices)
    
    return top_indices
