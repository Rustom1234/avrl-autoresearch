import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    """
    Selects documents based on word overlap with the target_sample.
    Prioritizes documents that share unique tokens with the target domain.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenize target sample to build a vocabulary of "important" words
    target_words = set()
    for s in target_sample:
        target_words.update(re.findall(r'\w+', s.lower()))
    
    # Score each document based on how many unique target words it contains
    # We also consider document length to ensure we don't just pick tiny snippets
    scores = []
    for i, doc in enumerate(docs):
        doc_words = set(re.findall(r'\w+', doc.lower()))
        overlap = len(doc_words.intersection(target_words))
        # Score is overlap count + a small bonus for length to ensure data density
        # but we prioritize overlap significantly.
        score = overlap + (len(doc_words) * 0.01)
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # To maintain diversity and ensure we don't over-train on a few perfect matches,
    # we take the top N documents and then fill the rest with a random selection.
    # We want to cover enough tokens to reach T, so we take a significant portion.
    num_docs = len(docs)
    top_count = min(num_docs // 2, 5000)
    
    selected_indices = [idx for score, idx in scores[:top_count]]
    
    # Add some random indices to ensure diversity
    remaining_indices = set(range(num_docs))
    remaining_indices.difference_update(selected_indices)
    
    # Pick a few more to ensure we have a healthy variety
    random_count = min(len(remaining_indices), 2000)
    random_list = list(remaining_indices)
    rng.shuffle(random_list)
    selected_indices.extend(random_list[:random_count])
    
    # The trainer cycles the list, so the order doesn't strictly matter 
    # as much as the composition, but we return the list of indices.
    return selected_indices
