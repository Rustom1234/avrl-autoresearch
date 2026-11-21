import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def get_embedding(text):
        tokens = enc.encode(text.lower())
        # Using a fixed size for embeddings to ensure consistent vector shapes
        # This allows for meaningful similarity comparison
        fixed_size = 100
        if len(tokens) < fixed_size:
            # Pad with a neutral token (e.g., 0 or the first token)
            tokens = tokens + [0] * (fixed_size - len(tokens))
        else:
            tokens = tokens[:fixed_size]
        
        # Represent the embedding as a normalized mean of the token IDs
        vec = np.array(tokens, dtype=np.float64)
        return vec / np.linalg.norm(vec)

    target_vectors = []
    for doc in target_sample:
        vec = get_embedding(doc)
        target_vectors.append(vec)
    
    if not target_vectors:
        return list(range(len(docs)))
    
    # Compute the centroid of the target sample
    target_avg_vec = np.mean(target_vectors, axis=0)
    target_avg_vec /= np.linalg.norm(target_avg_vec)
    
    scored_indices = []
    for i, doc in enumerate(docs):
        v = get_embedding(doc)
        
        # Cosine similarity: since vectors are normalized, it's the dot product
        similarity = np.dot(v, target_avg_vec)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top 50% of documents
    num_to_select = len(docs) // 2
    top_indices = [idx for score, idx in scored_indices[:num_to_select]]
    
    # Shuffle for diversity
    rng.shuffle(top_indices)
    
    return top_indices
