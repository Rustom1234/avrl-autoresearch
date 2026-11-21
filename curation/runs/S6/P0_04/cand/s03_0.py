"""Improved curator: estimates domain relevance by calculating the ratio of 
target vocabulary tokens in each document and sorts by high relevance."""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Create a set of all tokens present in the target sample to define the "domain"
    enc = tiktoken.get_encoding("gpt2")
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(enc.encode(text))
    
    # Calculate a relevance score for each document in the pool
    # Score = (count of tokens in target_tokens) / (total tokens in doc)
    scores = []
    for doc in docs:
        tokens = enc.encode(doc)
        if not tokens:
            scores.append((0.0, len(docs))) # Handle empty docs
            continue
            
        match_count = sum(1 for token in tokens if token in target_tokens)
        score = match_count / len(tokens)
        scores.append((score, len(docs)))
    
    # Create list of indices and sort by score descending
    # We use a stable sort to keep it deterministic for the same input
    indices = list(range(len(docs)))
    # Sort indices based on the score calculated for that index
    indices.sort(key=lambda i: scores[i][0], reverse=True)
    
    # While the prompt allows repeating indices, the instruction implies
    # iterating through the sorted list to fill T tokens.
    # Since we need an ordered list of indices to fill exactly T tokens:
    # If the sorted list of all indices provides enough tokens, we return it.
    # Because the trainer cycles the list if it's shorter than T, 
    # returning the sorted indices once is the standard approach.
    return indices
