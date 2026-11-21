"""Improved curator: Selects documents with the highest n-gram overlap to the target sample.
The trainer trains for T tokens. We prioritize high-quality, relevant documents.
"""
import os
import random
import collections
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def get_ngrams(text, n=3):
        tokens = enc.encode(text)
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    # Pre-calculate n-grams for target sample
    target_ngrams = []
    for doc in target_sample:
        target_ngrams.append(get_ngrams(doc))
    
    # Calculate average Jaccard-like overlap for each doc in the pool
    # We use a simplified overlap score to stay within time limits
    scores = []
    for i, doc in enumerate(docs):
        doc_ngrams = get_ngrams(doc)
        if not doc_ngrams:
            scores.append(-1.0)
            continue
            
        overlap_sum = 0
        for target in target_ngrams:
            intersection = len(doc_ngrams.intersection(target))
            union = len(doc_ngrams.union(target))
            if union > 0:
                overlap_sum += (intersection / union)
        
        scores.append(overlap_sum / len(target_ngrams))

    # Sort indices by score descending
    # We want to pick as many high-quality docs as possible to fill T tokens
    # Since repeats are allowed, and we want to maximize quality:
    indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    
    # To be safe and utilize the capacity, we return the best documents.
    # The trainer cycles the list if it's shorter than T.
    # We'll return a sufficiently large list of the best docs.
    # To ensure we don't just repeat one doc, we can take a reasonable slice.
    
    # Heuristic: return the top 1000 best docs to ensure variety and coverage
    # The trainer will cycle these if they don't reach T tokens.
    return indices[:1000]
