import os
import random
import collections
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken for accurate token counting as the trainer uses "gpt2"
    # The problem implies the trainer uses gpt2, and we need to reach T tokens.
    enc = tiktoken.get_encoding("gpt2")

    def get_vector(text):
        # Extract first 300 chars and count character frequencies
        chunk = text[:300].lower()
        counts = collections.Counter(chunk)
        # Define the alphabet used in the previous implementation
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789.,!?() '\""
        return np.array([counts.get(c, 0) for c in alphabet], dtype=np.float64)

    if not target_sample:
        return list(range(len(docs)))

    # Build profile from target sample
    target_vectors = [get_vector(doc) for doc in target_sample]
    avg_vec = np.mean(target_vectors, axis=0)
    
    mag_avg = np.linalg.norm(avg_vec)
    if mag_avg == 0:
        mag_avg = 1e-9

    # Score docs in pool
    scored_indices = []
    for i, doc in enumerate(docs):
        v = get_vector(doc)
        mag_v = np.linalg.norm(v)
        
        dot = np.dot(v, avg_vec)
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # We need enough documents to sum up to at least 2*T tokens
    # We calculate token counts for all docs to ensure we meet the criteria
    doc_token_counts = [len(enc.encode(doc)) for doc in docs]
    
    selected_indices = []
    current_tokens = 0
    
    # Iterate through sorted scores to pick documents until we hit 2*T
    # We prioritize the highest similarity scores.
    for score, idx in scored_indices:
        selected_indices.append(idx)
        current_tokens += doc_token_counts[idx]
        if current_tokens >= 2 * T:
            break
            
    # The problem asks to return indices. If we have fewer than 2*T but 
    # reached the end of the list, we have selected all docs sorted by similarity.
    # The trainer will cycle these.
    
    # To ensure we have a high quality set that meets the volume requirement, 
    # if we haven't hit 2*T because the pool is small, we just return all.
    if current_tokens < 2 * T and len(scored_indices) > 0:
        # This case is unlikely given "pool of about 2T tokens" but for safety:
        # return the best available.
        pass

    return selected_indices
