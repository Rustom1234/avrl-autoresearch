"""Curated curator: filters docs based on lexical overlap with the target_sample."""
import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenize target sample into a set of words for fast lookup
    # We use a simple regex to extract alphanumeric words to handle various punctuation
    target_words = set()
    for text in target_sample:
        target_words.update(re.findall(r'\w+', text.lower()))
    
    # If target_sample is empty, fallback to original behavior
    if not target_words:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # Calculate overlap scores
    scored_indices = []
    for i, doc in enumerate(docs):
        # Simple word overlap check
        doc_words = set(re.findall(r'\w+', doc.lower()))
        overlap = len(doc_words.intersection(target_words))
        # We only keep documents with at least one matching word to ensure relevance
        if overlap > 0:
            # Score is the proportion of the doc's words that match the target
            # This helps prioritize docs that are "closer" to the target domain
            score = overlap / max(len(doc_words), 1)
            scored_indices.append((score, i))
            
    # Sort by score descending and take the best indices
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # To ensure we have enough tokens, we take a larger subset or the whole pool
    # Since we need to fill T tokens and may need repeats, we take a generous amount
    # if the filtered list is small.
    filtered_indices = [idx for score, idx in scored_indices]
    
    # If filtering was too aggressive and we have very few docs, 
    # we fallback to the full list to ensure we don't run out of unique data 
    # before the cycling happens.
    if len(filtered_indices) < 10:
        filtered_indices = list(range(len(docs)))
    
    rng.shuffle(filtered_indices)
    return filtered_indices
