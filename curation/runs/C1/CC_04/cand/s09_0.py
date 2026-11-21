import os
import random
import collections
import numpy as np
from common import T
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        # Extract first 300 chars and find words/n-grams
        # Using a regex to extract words/alphanumerics as "tokens"
        chunk = text[:300].lower()
        # Simple word extraction
        return set(re.findall(r'\w+', chunk))

    # Build profile from target sample
    target_token_sets = [get_tokens(doc) for doc in target_sample]
    
    # Calculate a "representative" set of tokens from target sample
    # We take the union of all tokens in the target sample for the Jaccard base
    # Alternatively, a mean-like approach could be used, but Jaccard is requested.
    # Given 200 docs, the union might be too large/noisy. 
    # Let's use the most frequent tokens across the target sample to form the "ideal" set.
    all_target_tokens = collections.defaultdict(int)
    for s in target_token_sets:
        for token in s:
            all_target_tokens[token] += 1
    
    # Keep tokens that appear in at least 10% of the target sample (20 docs)
    # to define our "target domain" signature
    target_signature = {token for token, count in all_target_tokens.items() if count >= 20}
    
    if not target_signature:
        # Fallback to just any tokens if target_sample is too diverse/small
        target_signature = set().union(*target_token_sets) if target_token_sets else set()

    scored_indices = []
    for i, doc in enumerate(docs):
        doc_tokens = get_tokens(doc)
        if not doc_tokens or not target_signature:
            similarity = 0.0
        else:
            intersection = len(doc_tokens.intersection(target_signature))
            union = len(doc_tokens.union(target_signature))
            similarity = intersection / union if union > 0 else 0.0
        
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select indices
    # To ensure we have enough tokens and prioritize quality, 
    # we take a large enough portion of the pool.
    # The trainer cycles the list, so we want a good high-quality subset.
    top_count = max(len(docs) // 2, 1)
    top_indices = [idx for score, idx in scored_indices[:top_count]]
    
    # Shuffle the selected high-quality indices
    rng.shuffle(top_indices)
    
    return top_indices
