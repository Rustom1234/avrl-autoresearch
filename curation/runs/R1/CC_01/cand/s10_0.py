import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken
    enc = tiktoken.encoding.EncodingConverter.from_encoding_name("gpt2")
    # Note: standard usage is tiktoken.get_encoding("gpt2")
    # but since we are restricted to standard/scikit-learn/tiktoken, 
    # we use the standard loading method.
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if tiktoken version varies
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

    # Get the set of unique tokens from the target sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    scored_docs = []
    
    # Score each document
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        if len(tokens) < 100:
            continue
            
        # Calculate precision (percentage of tokens that appear in target_sample)
        matches = sum(1 for token in tokens if token in target_tokens)
        precision = matches / len(tokens)
        scored_docs.append((precision, i))
    
    # Sort by precision descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top 25% of indices
    num_to_keep = max(1, len(scored_docs) // 4)
    top_docs = scored_docs[:num_to_keep]
    
    # Extract indices and shuffle
    indices = [item[1] for item in top_docs]
    rng.shuffle(indices)
    
    return indices
