import os
import random
import collections
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_vector(text):
        # Extract first 300 chars and count character frequencies
        # Use a fixed alphabet to keep vectors consistent
        chunk = text[:300].lower()
        counts = collections.Counter(chunk)
        # We focus on standard alphanumeric/punctuation/space
        return [counts.get(c, 0) for c in "abcdefghijklmnopqrstuvwxyz0123456789.,!?() '\""]

    # Build profile from target sample
    target_vectors = [get_vector(doc) for doc in target_sample]
    if not target_vectors:
        return list(range(len(docs)))
        
    # Calculate "average" vector (centroid)
    avg_vec = np.mean(target_vectors, axis=0)
    
    # Score docs in pool
    pool_indices = []
    # We want to find the best candidates to fill the T tokens
    # Score = Cosine Similarity = (A.B) / (|A|*|B|)
    scored_indices = []
    
    for i, doc in enumerate(docs):
        v = get_vector(doc)
        # Simple dot product for similarity
        dot = np.dot(v, avg_vec)
        # Norms
        mag_v = np.linalg.norm(v)
        mag_avg = np.linalg.norm(avg_vec)
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 50% of documents to ensure variety and sufficient volume
    # but prioritize high similarity
    top_indices = [idx for score, idx in scored_indices[:len(docs)//2]]
    
    # Shuffle the top candidates to maintain some diversity
    rng.shuffle(top_indices)
    
    # Ensure we return enough to cover T (the trainer cycles if short)
    # We return the shuffled high-quality indices.
    return top_indices
