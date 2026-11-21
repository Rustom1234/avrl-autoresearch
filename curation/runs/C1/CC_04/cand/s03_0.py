import os
import random
import collections
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.get_encoding("gpt2")

    def get_embedding(text):
        # Use a larger window and more inclusive set for character frequencies
        chunk = text[:500].lower()
        counts = collections.Counter(chunk)
        # Using a set of characters present in typical text to form a fixed-dim vector
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789.,!?() \"'_-+=\\/\\,.;:?@#$%^&*()[]{}|<>>"
        return np.array([counts.get(c, 0) for c in alphabet], dtype=np.float32)

    if not target_sample:
        return list(range(min(len(docs), 100)))

    # Calculate centroid of target sample
    target_vectors = [get_embedding(doc) for doc in target_sample]
    centroid = np.mean(target_vectors, axis=0)
    centroid_norm = np.linalg.norm(centroid)

    scored_docs = []
    for i, doc in enumerate(docs):
        vec = get_embedding(doc)
        vec_norm = np.linalg.norm(vec)
        # Cosine Similarity
        sim = np.dot(vec, centroid) / (vec_norm * centroid_norm + 1e-9)
        # Calculate token count for length requirement
        tokens = encoder.encode(doc)
        scored_docs.append({
            'index': i,
            'sim': sim,
            'len': len(tokens),
            'doc': doc
        })

    # Sort by similarity descending
    scored_docs.sort(key=lambda x: x['sim'], reverse=True)

    # Soft-Max style weighting / Probability distribution
    # Shift to positive for softmax
    max_sim = max(d['sim'] for d in scored_docs)
    exp_sims = np.exp(np.array([d['sim'] - max_sim for d in scored_docs]))
    probs = exp_sims / np.sum(exp_sims)

    # Selection process: We need at least 2*T tokens.
    # We use the probability distribution to sample while prioritizing high similarity.
    # To satisfy the "Top indices" requirement while meeting the 2T volume, 
    # we greedily take top candidates until 2T is reached.
    
    selected_indices = []
    total_tokens = 0
    target_tokens = 2 * T
    
    # Iterate through sorted docs and pick if similarity is above a dynamic threshold
    # or until we hit the token quota.
    for item in scored_docs:
        selected_indices.append(item['index'])
        total_tokens += item['len']
        if total_tokens >= target_tokens:
            break
            
    # If the pool is exhausted and we still need tokens, fill with random high-sim docs
    if total_tokens < target_tokens:
        remaining = target_tokens - total_tokens
        # This part is rarely reached given the 2T pool size, but handles edge cases
        indices_to_add = rng.sample(range(len(docs)), min(len(docs) - len(selected_indices), 100))
        selected_indices.extend(indices_to_add)

    return selected_indices
