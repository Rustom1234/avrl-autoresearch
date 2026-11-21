import os
import random
import re
from collections import Counter
import numpy as np

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use a simple tokenization to stay within time limits
    def get_tokens(text):
        return text.split()

    # 1. Fast Domain Filtering
    # Instead of heavy bigrams, use a keyword/structure heuristic to find "clean" text
    candidate_indices = []
    # Target words/characteristics (generalized to identify high-quality prose)
    for i, doc in enumerate(docs):
        if len(doc) < 100: continue
        # Penalize heavy code or junk patterns
        if any(x in doc for x in ['import ', 'def ', 'public static', '<?xml', '<html>']):
            continue
        candidate_indices.append(i)

    # 2. Relevance Scoring via Word Co-occurrence
    # Identify common words in the target domain
    target_words = set()
    for ts in target_sample:
        target_words.update(set(t.lower() for t in get_tokens(ts) if len(t) > 3))
    
    # Calculate overlap scores
    scored_indices = []
    for idx in candidate_indices:
        doc_words = set(t.lower() for t in get_tokens(docs[idx]) if len(t) > 3)
        overlap = len(doc_words.intersection(target_words))
        if overlap > 0:
            # Score is normalized by document length to prefer dense relevance
            score = overlap / (len(doc_words) + 1)
            scored_indices.append((idx, score))
            
    # Sort by relevance score descending
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Deduplication and Selection
    final_indices = []
    seen_content = set()
    
    # We want enough tokens to hit T = 9,830,400. 
    # A safe heuristic: pick a large enough number of high-quality docs.
    # We stop early if we reach a reasonable proxy for T, but 
    # the prompt says return a list of indices (repeats allowed).
    # We will pick the top 1000 unique high-quality docs to ensure variety.
    for idx, score in scored_indices:
        doc_chunk = docs[idx][:200].strip().lower()
        # Basic hashing to avoid exact duplicates
        h = hash(doc_chunk)
        if h not in seen_content:
            seen_content.add(h)
            final_indices.append(idx)
        if len(final_indices) >= 2000:
            break

    # If we didn't find enough, fill with other candidates
    if len(final_indices) < 1:
        final_indices = candidate_indices[:min(len(candidate_indices), 1000)]

    # Ensure the list is not empty
    if not final_indices:
        return [0] if docs else []

    # Shuffle to provide diversity in training
    rng.shuffle(final_indices)
    return final_indices
