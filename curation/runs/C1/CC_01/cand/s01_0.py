import os
import random
from collections import Counter

def select(docs, target_sample):
    """
    Curates the document pool by prioritizing documents that share 
    significant content (common substrings) with the target sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Extract key features from target_sample to build a profile
    # Using a set of words/substrings longer than 4 chars as a heuristic
    target_features = set()
    for doc in target_sample:
        # Simple tokenization/filtering
        words = [w.strip(".,!?;:()\"'") for w in doc.split() if len(w) > 4]
        target_features.update(words)
        
    # Score documents in the pool based on overlap with target_features
    doc_scores = []
    for i, doc in enumerate(docs):
        score = 0
        # Simple overlap counting
        words = [w.strip(".,!?;:()\"'") for w in doc.split() if len(w) > 4]
        for w in words:
            if w in target_features:
                score += 1
        doc_scores.append((score, i))
    
    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    sorted_indices = [idx for score, idx in doc_scores]
    
    # While we want to prioritize high-quality docs, we might need to 
    # fill the space. Since the trainer cycles the list to reach T,
    # the ordering of the high-score docs first is the primary goal.
    # We return the sorted indices.
    return sorted_indices
