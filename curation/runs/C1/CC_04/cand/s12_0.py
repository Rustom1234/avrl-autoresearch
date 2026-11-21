import os
import random
import collections
import numpy as np
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def is_encyclopedic(text):
        # Heuristic for "Wikipedia-like" formatting:
        # 1. Starts with a Title in quotes, brackets, or capitalized first word
        # 2. Followed by a definition (e.g., "is a", "refers to", "defined as")
        # 3. Contains specific punctuation or structured layout
        title_match = re.search(r'^(\[[^\]]+\]|"[^"]+"|\w+)\s+([is|refers|defined|a|an|the])', text[:100], re.IGNORECASE)
        # Or more simply, presence of bracketed terms or title-like structures
        has_structure = bool(re.search(r'\[[^\]]+\]|"\w+"', text[:150]))
        has_definition = bool(re.search(r'\b(is|are|refers to|refers to|defined as|a type of)\b', text[:150], re.IGNORECASE))
        return has_structure and has_definition

    def get_vector(text):
        chunk = text[:300].lower()
        counts = collections.Counter(chunk)
        # Standard alphabet + common punctuation
        chars = "abcdefghijklmnopqrstuvwxyz0123456789.,!?() '\""
        return np.array([counts.get(c, 0) for c in chars], dtype=np.float64)

    # Identify encyclopedic documents in target_sample
    target_encyclopedic_vectors = []
    for doc in target_sample:
        if is_encyclopedic(doc):
            target_encyclopedic_vectors.append(get_vector(doc))
            
    if not target_encyclopedic_vectors:
        # Fallback to standard similarity if no encyclopedic target found
        target_vectors = [get_vector(doc) for doc in target_sample]
    else:
        target_vectors = target_encyclopedic_vectors
        
    avg_vec = np.mean(target_vectors, axis=0)
    
    scored_indices = []
    for i, doc in enumerate(docs):
        # Priority weight: check if document matches the encyclopedic style
        is_style_match = is_encyclopedic(doc)
        
        v = get_vector(doc)
        dot = np.dot(v, avg_vec)
        mag_v = np.linalg.norm(v)
        mag_avg = np.linalg.norm(avg_vec)
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        
        # Boost score if it matches the structural heuristic
        score = similarity * (1.5 if is_style_match else 1.0)
        scored_indices.append((score, i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select top candidates
    # Use a larger pool to ensure we have enough tokens for T
    # but keep the highest quality first.
    # Since the trainer cycles, we want high quality first.
    num_to_keep = min(len(docs), 2000) 
    top_indices = [idx for score, idx in scored_indices[:num_to_keep]]
    
    # Re-shuffle to prevent the model from seeing the same order every time
    # if there were multiple runs, though here the trainer is fixed.
    # We don't shuffle fully because we want the highest quality first.
    # We keep them ordered by score but could slightly jitter.
    
    return top_indices
