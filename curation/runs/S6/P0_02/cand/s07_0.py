"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re

def select(docs, target_sample):
    # Pre-filter: remove docs containing 'DSO' or starting with '<'
    # Also implement keyword-based confidence filter for spam/repetitive patterns
    filtered_indices = []
    
    # Patterns for spam/catalogs/repetitive content
    spam_patterns = [
        r"(?i)click\s+here",
        r"(?i)\w+\s+=\s+\$\d+",
        r"(?i)\w+\s+=\s+\d+",
        r"(\b\w+\b\s*){10,}" # potential for long repetitive lists
    ]
    
    # Pre-compile patterns for speed
    compiled_patterns = [re.compile(p) for p in spam_patterns]

    for i, doc in enumerate(docs):
        # Basic noise filters
        if "DSO" in doc or doc.startswith('<'):
            continue
            
        # Confidence filter: check first 500 chars for spam signatures
        preview = doc[:500]
        is_spam = False
        
        # Count occurrences of specific spam phrases
        if (preview.lower().count("click here") > 1):
            is_spam = True
        else:
            # Check regex patterns
            for pattern in compiled_patterns:
                if pattern.search(preview):
                    is_spam = True
                    break
        
        # Check for repetitive sentence structures (highly repetitive words in first 500)
        words = re.findall(r'\w+', preview.lower())
        if len(words) > 20:
            # Simple heuristic: if a word repeats too frequently in a short span
            counts = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
                if counts[w] > 10: # Single word repeating 10+ times
                    is_spam = True
                    break
        
        if not is_spam:
            filtered_indices.append(i)
    
    # Shuffle the filtered indices to ensure variety
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(filtered_indices)
    
    return filtered_indices
