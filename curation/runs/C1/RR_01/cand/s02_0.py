"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import collections
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Simple tokenizer to extract words
    def get_words(text):
        return re.findall(r'\w+', text.lower())

    # Count word frequencies in target_sample
    target_counts = collections.Counter()
    target_words_all = []
    for doc in target_sample:
        words = get_words(doc)
        target_words_all.extend(words)
        target_counts.update(words)
    
    # Count word frequencies in the entire pool
    pool_counts = collections.Counter()
    for doc in docs:
        words = get_words(doc)
        pool_counts.update(words)
        
    # Calculate TF-IDF inspired scores
    # Score = Sum of (TargetFreq[w] / (1 + PoolFreq[w])) for w in doc
    doc_scores = []
    for i, doc in enumerate(docs):
        words = get_words(doc)
        score = 0.0
        # Only consider words that exist in the target sample to focus on domain relevance
        # Use a set for words in the current doc to avoid overcounting long repetitive docs
        unique_doc_words = set(words)
        for w in unique_doc_words:
            if w in target_counts:
                # Weight: high frequency in target, low frequency in pool
                # We use a simple inverse frequency for the pool
                pool_freq = pool_counts.get(w, 0)
                score += target_counts[w] / (1.0 + pool_freq)
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the indices in sorted order
    # Since we need to fill T tokens, and we can repeat, 
    # we return the best indices.
    sorted_indices = [item[1] for item in doc_scores]
    
    # Because we can repeat and the trainer cycles if list < T,
    # we simply return the ranked list.
    return sorted_indices
