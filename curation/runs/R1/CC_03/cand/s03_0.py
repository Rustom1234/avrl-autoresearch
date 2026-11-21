import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode the first 100 tokens of each document in the target_sample
    # and build a collective vocabulary (set of unique tokens)
    target_vocab = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        target_vocab.update(tokens[:100])
    
    # Calculate Jaccard similarity for each document in the pool
    # Similarity = (Intersection size) / (Union size)
    # Note: Jaccard score of 0 is handled by sorting.
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        # We only look at the first 100 tokens of the doc to match instructions
        # for "similarity of its first 100 tokens"
        doc_tokens_100 = set(enc.encode(doc)[:100])
        
        intersection = len(doc_tokens_100.intersection(target_vocab))
        union = len(doc_tokens_100.union(target_vocab))
        
        score = intersection / union if union > 0 else 0.0
        scores.append((score, i))
    
    # Sort by score descending, then by original index to keep it stable
    # scores = [(score, index), ...]
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Return only the indices
    indices = [item[1] for item in scores]
    
    # The trainer cycles the list, but we should ensure we provide 
    # a sufficient number of indices. However, the prompt says 
    # "return them", and the trainer cycles what is provided.
    # To maximize quality, we return the sorted unique indices.
    return indices
