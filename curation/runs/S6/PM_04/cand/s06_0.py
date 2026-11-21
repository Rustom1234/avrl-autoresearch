import os
import random
import re
import numpy as np
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Tokenize target sample and build TF-IDF-like frequency maps
    target_tokens = []
    for s in target_sample:
        target_tokens.extend(enc.encode(s))
    
    target_counts = Counter(target_tokens)
    # Total unique tokens in target to normalize
    total_unique_target = len(target_counts)
    
    scored_docs = []

    # Markers for structural score
    markers = [r'@-@', r'@,@', r'@.@', r',', r'\.']
    
    for i, doc in enumerate(docs):
        if len(doc) < 10:
            continue
            
        # Structural Score: count specific punctuation/markers per 1000 chars
        # This rewards text that follows standard linguistic structures
        struct_score = 0
        for m in markers:
            struct_score += len(re.findall(m, doc))
        
        # Normalized by length (per 1000 characters)
        normalized_struct = (struct_score / len(doc)) * 1000

        # Relevance Score: TF-IDF style logic
        # We count how many target tokens appear in the document
        doc_tokens = enc.encode(doc)
        doc_counts = Counter(doc_tokens)
        relevance_score = 0
        for token, count in doc_counts.items():
            if token in target_counts:
                # Reward presence of target tokens, scaled by their target frequency
                # (Using a simple weight to prioritize common-but-relevant terms)
                relevance_score += count * (target_counts[token] / total_unique_target)

        # Combine scores
        # Relevance is prioritized, but structural integrity acts as a multiplier/gate
        # Documents with zero relevance or very low structure are penalized
        total_score = relevance_score + (normalized_struct * 0.5)
        
        # Only include docs that show some signal
        if total_score > 0:
            scored_docs.append((i, total_score))

    # Sort by score descending to pick the best documents first
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top N or the majority of available docs to ensure we have enough content
    # But ensure we don't exceed the total available docs
    limit = min(len(docs), 2000) # Arbitrary limit to keep the selection manageable
    selected_indices = [item[0] for item in scored_docs[:limit]]

    if not selected_indices:
        # Fallback to shuffled indices if no docs scored
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices

    # Shuffle the selected indices to avoid bias in the training sequence
    rng.shuffle(selected_indices)
    return selected_indices
