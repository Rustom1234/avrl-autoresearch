import os
import random
import collections
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use the gpt2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    def get_token_vector(text):
        # Take first 300 chars and encode. 
        # Removed the 'errors' keyword argument which caused the TypeError.
        chunk = text[:300]
        tokens = enc.encode(chunk)
        counts = collections.Counter(tokens)
        return counts

    # Build profile from target sample
    target_counts_list = []
    for doc in target_sample:
        target_counts_list.append(get_token_vector(doc))
        
    if not target_counts_list:
        return list(range(len(docs)))

    # Calculate "average" vector (centroid)
    combined_counts = collections.Counter()
    for counts in target_counts_list:
        combined_counts.update(counts)
    
    num_targets = len(target_sample)
    avg_vec = {k: v / num_targets for k, v in combined_counts.items()}

    # Score docs in pool
    scored_indices = []
    for i, doc in enumerate(docs):
        v_counts = get_token_vector(doc)
        
        # Dot product
        dot = 0.0
        for token_id, count in v_counts.items():
            dot += count * avg_vec.get(token_id, 0.0)
            
        # Norms
        mag_v = np.linalg.norm(list(v_counts.values()))
        
        # Correct Mag Avg for the centroid
        mag_avg = np.sqrt(sum(v**2 for v in avg_vec.values()))
        
        similarity = dot / (mag_v * mag_avg + 1e-9)
        scored_indices.append((similarity, i))
    
    # Sort by similarity descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 50% to maintain variety but prioritize high similarity
    # We take the top half of the sorted list.
    num_to_keep = len(docs) // 2
    top_indices = [idx for score, idx in scored_indices[:num_to_keep]]
    
    # Shuffle the top indices to ensure the model sees high-quality 
    # data distributed throughout the start of the stream.
    rng.shuffle(top_indices)
    
    return top_indices
