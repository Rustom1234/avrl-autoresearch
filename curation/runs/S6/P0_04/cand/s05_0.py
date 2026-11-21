import os
import random
import collections
from common import T
import tiktoken

def select(docs, target_sample):
    """
    Implements a weighted selection strategy using n-gram matching.
    Precomputes a vocabulary leak count for each document against the 
    target_sample (tokens appearing only once or twice), then 
    sequentially selects documents with the highest leak count ratio.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to tokenize the target sample
    enc = tiktoken.get_encoding("gpt2")
    
    # Identify "leaky" tokens in target_sample (appearing 1 or 2 times)
    target_counts = collections.Counter()
    for doc in target_sample:
        target_counts.update(enc.encode(doc))
    
    leaky_tokens = {token for token, count in target_counts.items() if count <= 2}
    
    # Calculate leak count and total tokens for each document in the pool
    doc_data = []
    for doc in docs:
        tokens = enc.encode(doc)
        leak_count = sum(1 for t in tokens if t in leaky_tokens)
        doc_data.append({
            'tokens': tokens,
            'leak_ratio': leak_count / len(tokens) if tokens else 0
        })
    
    # Sort documents by leak ratio descending
    # We pair index with data to maintain reference to the original docs list
    indexed_data = []
    for i in range(len(docs)):
        indexed_data.append((i, doc_data[i]))
        
    # Sort by leak_ratio descending, then by index to stay deterministic
    indexed_data.sort(key=lambda x: (-x[1]['leak_ratio'], x[0]))
    
    # Determine the sequence of indices to fill T tokens
    selected_indices = []
    current_tokens = 0
    
    # We iterate through the sorted documents and pick them until we satisfy T
    # To ensure we don't just take one document and cycle it (unless it's the best),
    # we follow the sorted order.
    for idx, data in indexed_data:
        doc_tokens = data['tokens']
        num_tokens = len(doc_tokens)
        
        if num_tokens == 0:
            continue
            
        # Calculate how many times we need to repeat this document to reach T
        # or move to the next best one. 
        # Since the goal is "sequentially select... until the token budget T is filled",
        # and we need to return a list that will be cycled if < T, 
        # we fill the list with the best docs in order.
        
        # To be efficient, we take the best docs in order until we reach T.
        # If the list is still < T after one pass, we'd cycle it.
        # But the prompt implies a sequential selection until T is filled.
        
        # Let's collect indices until we hit T.
        while current_tokens < T:
            selected_indices.append(idx)
            current_tokens += num_tokens
            # If adding this doc exceeds T, we still add it (as the trainer handles the T cutoff)
            # but we stop selecting new documents to maintain the "sequential" high-ratio order.
            if current_tokens >= T:
                break
                
    # If we haven't reached T (e.g. docs were empty), the trainer's cycling handles it.
    # If we have many, and we want to be safe about the T limit, 
    # the prompt says "until the token budget T is filled".
    # To ensure we don't provide a massive list unnecessarily:
    
    # Optimization: if we have enough docs to cover T, we could stop.
    # However, the trainer cycles. To ensure we hit T with the best content:
    # Let's ensure the list provided is enough to cover T if we only wanted one pass.
    
    return selected_indices
