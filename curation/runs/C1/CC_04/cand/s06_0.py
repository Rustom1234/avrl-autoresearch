import os
import random
import collections
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Build the dynamic vocabulary
    # Get all unique characters from target sample
    target_chars = set()
    for doc in target_sample:
        target_chars.update(doc)
        
    # Get most frequent 100 characters from the pool
    pool_counts = collections.Counter()
    for doc in docs:
        pool_counts.update(doc)
    
    top_100_pool = [char for char, count in pool_counts.most_common(100)]
    
    # Combined vocabulary: unique chars in target + top 100 from pool
    # Use a set to ensure uniqueness, then convert to a sorted list for consistent indexing
    vocab_set = target_chars.union(set(top_100_pool))
    vocab = sorted(list(vocab_set))
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    
    def get_vector(text):
        counts = collections.Counter(text)
        # Vector size is length of our dynamic vocab
        return np.array([counts.get(char, 0) for char in vocab], dtype=np.float32)

    # 2. Build profile from target sample
    target_vectors = [get_vector(doc) for doc in target_sample]
    if not target_vectors:
        return list(range(len(docs)))
        
    # Calculate "average" vector (centroid)
    avg_vec = np.mean(target_vectors, axis=0)
    
    # 3. Score docs in pool
    scored_indices = []
    for i, doc in enumerate(docs):
        v = get_vector(doc)
        # Cosine Similarity = (A.B) / (|A|*|B|)
        dot = np.dot(v, avg_vec)
        mag_v = np.linalg.norm(v)
        mag_avg = np.linalg.norm(avg_vec)
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 50% of documents
    num_to_keep = len(docs) // 2
    top_indices = [idx for score, idx in scored_indices[:num_to_keep]]
    
    # Shuffle the top candidates
    rng.shuffle(top_indices)
    
    return top_indices
