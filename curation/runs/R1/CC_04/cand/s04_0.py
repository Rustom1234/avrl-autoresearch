"""Default curator ("weighted_jaccard"): ranks docs by Jaccard similarity to the target_sample word set, 
weighted by proximity to the average target_sample length.
"""
import os
import random
import re
import numpy as np
from collections import Counter
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    def get_words(text):
        # Simple word extraction to match Jaccard logic on "words"
        return set(re.findall(r'\w+', text.lower()))

    # Calculate target stats
    target_word_set = set()
    target_lengths = []
    for s in target_sample:
        target_word_set.update(get_words(s))
        target_lengths.append(len(enc.encode(s)))
    
    avg_target_len = sum(target_lengths) / len(target_lengths) if target_lengths else 0
    
    # Score documents
    scored_docs = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        doc_len = len(tokens)
        doc_words = get_words(doc)
        
        # Jaccard Similarity
        intersection = len(doc_words.intersection(target_word_set))
        union = len(doc_words.union(target_word_set))
        jaccard = intersection / union if union > 0 else 0
        
        # Length proximity weight: 1 / (1 + abs(doc_len - avg_target_len))
        # We use a small epsilon or handle zero to ensure it doesn't explode
        proximity = 1.0 / (1.0 + abs(doc_len - avg_target_len))
        
        # Score is product of similarity and proximity
        score = jaccard * proximity
        scored_docs.append((score, i))
    
    # Sort indices by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    best_indices = [item[1] for item in scored_docs]
    
    # Since the trainer cycles the list, we can return the sorted indices.
    # The trainer will cycle through these until T tokens are reached.
    return best_indices
