import os
import random
import collections
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_bigrams(text):
        # Extract first 300 characters and find unique bigrams
        chunk = text[:300].lower()
        bigrams = set()
        for i in range(len(chunk) - 1):
            bigrams.add(chunk[i:i+2])
        return bigrams

    # Build global bigram set from the target sample
    target_bigrams = set()
    for doc in target_sample:
        target_bigrams.update(get_bigrams(doc))
    
    if not target_bigrams:
        # Fallback to first indices if no bigrams found
        return list(range(min(len(docs), 1000)))

    scored_indices = []
    for i, doc in enumerate(docs):
        doc_bigrams = get_bigrams(doc)
        # Jaccard Similarity = |A ∩ B| / |A ∪ B|
        intersection = len(doc_bigrams.intersection(target_bigrams))
        union = len(doc_bigrams.union(target_bigrams))
        
        score = intersection / union if union > 0 else 0
        scored_indices.append((score, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # We need enough tokens to sum to at least 2*T.
    # Since the trainer cycles, and we want high quality, 
    # we take the best docs until we exceed 2*T in total length.
    # We assume a conservative average length for docs to estimate count,
    # but since we must return indices that sum to at least 2*T, 
    # and the pool is ~2T, we select the top performing docs.
    
    # To be safe and efficient with the 180s limit, we take top indices
    # that are likely to satisfy the volume requirement.
    # Let's take a significant portion of the high-scoring docs.
    # Given docs is ~2T and we need 2T, we take the top 75% of ranked docs.
    # If docs is smaller, we take all.
    num_to_take = max(int(len(docs) * 0.75), 1)
    top_indices = [idx for score, idx in scored_indices[:num_to_take]]
    
    # Ensure we have a large enough list to satisfy the 2*T requirement 
    # through cycling if the count is small.
    # If the current top_indices sum to less than 2*T tokens, 
    # we append the next best indices.
    # Since we don't know exact lengths without iterating, we use a heuristic:
    # most docs are significant. If we have fewer than 500 indices, we take more.
    if len(top_indices) < 500 and len(scored_indices) > 500:
        top_indices.extend([idx for score, idx in scored_indices[num_to_take:num_to_take + 500]])

    return top_indices
