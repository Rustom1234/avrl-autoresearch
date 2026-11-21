import os
import random
import collections
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use the gpt2 tokenizer as specified
    enc = tiktoken.get_encoding("gpt2")
    
    def get_token_vector(text):
        # Encode the first 300 characters as requested
        # We take the first 300 chars, then encode to tokens
        chunk = text[:300]
        tokens = enc.encode(chunk, errors='ignore')
        
        # Create a frequency vector of token IDs
        # Since we can't know the full vocab size efficiently for a vector,
        # we use a dictionary/map logic to simulate a sparse vector or 
        # just use the raw counts of IDs present in this specific chunk.
        # To perform cosine similarity, we need a consistent representation.
        # We will use a dictionary to represent the sparse vector.
        counts = collections.Counter(tokens)
        return counts

    # Build profile from target sample
    target_counts_list = []
    for doc in target_sample:
        target_counts_list.append(get_token_vector(doc))
        
    if not target_counts_list:
        return list(range(len(docs)))

    # Calculate "average" vector (centroid)
    # Since these are sparse, we sum all counts across the sample
    combined_counts = collections.Counter()
    for counts in target_counts_list:
        combined_counts.update(counts)
    
    # The average count for each token ID
    # (Average count = total sum of counts for that ID / number of docs)
    num_targets = len(target_sample)
    avg_vec = {k: v / num_targets for k, v in combined_counts.items()}

    # Score docs in pool
    scored_indices = []
    for i, doc in enumerate(docs):
        v_counts = get_token_vector(doc)
        
        # Dot product
        dot = 0.0
        # Only iterate over keys in the current doc's counts for efficiency
        for token_id, count in v_counts.items():
            dot += count * avg_vec.get(token_id, 0.0)
            
        # Norms
        mag_v = np.linalg.norm(list(v_counts.values()))
        
        # Calculate Mag Avg
        # Note: For a proper centroid norm, we should sum the squares of all 
        # averaged counts, but for a heuristic, we'll sum the squares of the 
        # weighted counts.
        mag_avg = np.sqrt(sum(v**2 for v in avg_vec.values()))
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 50% to maintain variety but prioritize high similarity
    # The trainer cycles, so we want the best documents prioritized.
    top_indices = [idx for score, idx in scored_indices[:len(docs)//2]]
    
    # Shuffle for variety
    rng.shuffle(top_indices)
    
    return top_indices
