import os
import random
import re
from collections import Counter
import numpy as np

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-calculate target keywords efficiently
    target_keywords = Counter()
    # Using a regex to find words and only keeping a set of unique words to speed up
    # we avoid over-weighting specific words by using a set if they are too frequent
    for sample in target_sample:
        words = re.findall(r'[a-zA-Z]{5,}', sample.lower())
        for w in words:
            target_keywords[w] += 1

    # We need to process docs quickly. 
    # Instead of full string searches in a loop, we'll sample the pool 
    # to build a representative vocabulary or use a faster scoring method.
    
    # Optimization: Use a subset of docs for scoring to save time
    # or perform a single pass and use efficient string membership.
    
    # Filter for high quality (minimal logic to stay under 180s)
    # We skip heavy regex/scoring on the whole pool if it's too large.
    valid_indices = []
    limit = 5000 # Limit processing to a manageable subset for speed
    
    # Fast scoring
    scored_docs = []
    
    # We'll iterate through all docs, but use a very fast scoring method.
    # To handle the time limit, we process the full list but keep the logic lean.
    for i, doc in enumerate(docs):
        # Basic Junk Filter
        if len(doc) < 100:
            continue
            
        score = 0
        # Only check a subset of high-frequency target keywords to save time
        # Or iterate keywords and check doc membership
        # To be even faster, we can check if any word from the top keywords is in doc
        doc_lower = doc.lower()
        for word, weight in target_keywords.items():
            if word in doc_lower:
                score += weight
        
        scored_docs.append((score, i))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the best documents
    # We want to ensure we have enough tokens. The trainer cycles the list.
    # We'll take a significant portion of the pool.
    num_to_take = min(len(scored_docs), 2000)
    selected_indices = [idx for score, idx in scored_docs[:num_to_take]]
    
    # Fill the rest with random indices from the original pool if needed
    # but usually, the top scores are what we want.
    # To ensure we hit the T token requirement efficiently, we provide a large list.
    all_indices = list(range(len(docs)))
    remaining = list(set(all_indices) - set(selected_indices))
    rng.shuffle(remaining)
    
    # Combine and shuffle to provide variety
    final_selection = selected_indices + remaining[:num_to_take]
    rng.shuffle(final_selection)
    
    return final_selection
