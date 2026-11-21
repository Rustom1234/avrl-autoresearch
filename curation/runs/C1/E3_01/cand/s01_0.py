import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    """
    Curates documents by prioritizing those with high token overlap 
    with the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target sample to build a reference vocabulary/frequency
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(enc.encode(text))
    
    # Calculate scores for each document in the pool
    # Score = (Number of shared tokens) / (Total tokens in doc)
    # This prioritizes dense, relevant content over long, noisy content.
    doc_scores = []
    for doc in docs:
        tokens = enc.encode(doc)
        if not tokens:
            doc_scores.append((0.0, 0))
            continue
            
        shared_count = sum(1 for t in tokens if t in target_tokens)
        score = shared_count / len(tokens)
        doc_scores.append((score, len(tokens)))
    
    # Pair indices with scores and sort by score descending
    # We include length as a tie-breaker to prefer longer (more informative) docs
    indexed_scores = []
    for i in range(len(docs)):
        score, length = doc_scores[i]
        indexed_scores.append((score, length, i))
        
    # Sort: Primary key score (desc), Secondary key length (desc)
    indexed_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Get the indices of the top performing documents
    # Since we can repeat indices and we want to maximize information,
    # we pick the top documents until we fill the requirements.
    # However, the simplest effective strategy is to return the sorted 
    # indices to ensure the model sees the highest quality data first.
    sorted_indices = [item[2] for item in indexed_scores]
    
    return sorted_indices
