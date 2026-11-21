"""Default curator ("refined"): filters out short or junk documents before selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken

def select(docs, target_sample):
    encoder = tiktoken.get_encoding("gpt2")
    
    # Define junk patterns
    junk_patterns = ["discount code", "click here"]
    
    valid_indices = []
    for i, doc in enumerate(docs):
        # Tokenize to check length and content
        tokens = encoder.encode(doc)
        
        # Filter 1: Extremely short documents (less than 100 tokens)
        if len(tokens) < 100:
            continue
            
        # Filter 2: Repetitive junk patterns
        # If a document contains any junk pattern more than twice, it's junk
        is_junk = False
        doc_lower = doc.lower()
        for pattern in junk_patterns:
            if doc_lower.count(pattern) > 2:
                is_junk = True
                break
        
        if is_junk:
            continue
            
        valid_indices.append(i)
    
    # If everything was filtered out (unlikely), fallback to all
    if not valid_indices:
        valid_indices = list(range(len(docs)))

    # Shuffle the valid indices
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(valid_indices)
    
    return valid_indices
