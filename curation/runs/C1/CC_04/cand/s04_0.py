import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to encode the documents
    # The prompt specifies "gpt2" encoding
    enc = tiktoken.get_encoding("gpt2")
    
    def get_embedding(text):
        # Encode and take the first 100 tokens
        tokens = enc.encode(text.lower())
        tokens = tokens[:100]
        if not tokens:
            return np.array([0.0])
        # Create a simple mean embedding from the token IDs
        # Since we don't have the actual model weights, 
        # we use the token IDs as the basis for the vector
        # to represent the "embedding" space requested.
        vec = np.array(tokens, dtype=np.float64)
        return vec

    # Build target profile
    target_vectors = [get_embedding(doc) for doc in target_sample]
    if not target_vectors:
        return list(range(len(docs)))
    
    # Calculate the mean embedding of the target sample
    # Using a mean over the token positions for the target sample
    target_avg_vec = np.mean(target_vectors, axis=0)
    
    # Score docs in pool
    scored_indices = []
    for i, doc in enumerate(docs):
        v = get_embedding(doc)
        
        # Calculate Cosine Similarity
        dot = np.dot(v, target_avg_vec)
        mag_v = np.linalg.norm(v)
        mag_avg = np.linalg.norm(target_avg_vec)
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top half of the documents based on similarity
    # This provides the highest quality material for the training budget
    top_indices = [idx for score, idx in scored_indices[:len(docs)//2]]
    
    # Shuffle for diversity while maintaining the high-quality set
    rng.shuffle(top_indices)
    
    return top_indices
