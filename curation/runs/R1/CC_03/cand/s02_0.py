import os
import random
from collections import Counter

def select(docs, target_sample):
    """
    Scores documents based on the ratio of their shared vocabulary with 
    the target_sample and returns indices sorted by score descending.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Build a set of all unique words in the target sample
    target_vocab = set()
    for doc in target_sample:
        # Basic split to find words (keeping it simple for performance)
        words = doc.lower().split()
        target_vocab.update(words)
    
    # Calculate scores for each document in the pool
    # Score = (unique words in doc that are also in target_vocab) / (total unique words in doc)
    scores = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        if not words:
            scores.append((0.0, i))
            continue
            
        doc_words = set(words)
        shared = len(doc_words.intersection(target_vocab))
        score = shared / len(doc_words) if len(doc_words) > 0 else 0.0
        scores.append((score, i))
    
    # Sort by score descending, then by original index to maintain stability
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Return only the indices
    return [item[1] for item in scores]
