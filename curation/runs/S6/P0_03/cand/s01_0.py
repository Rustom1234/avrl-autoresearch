"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

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
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    filtered_indices = []
    
    for i, doc in enumerate(docs):
        # Quick check to avoid tokenizing massive junk if possible
        # but we must check the first 200 tokens as per instructions.
        tokens = encoder.encode(doc)
        relevant_tokens = tokens[:200]
        
        is_junk = False
        
        # Heuristics:
        # 1. Mathematical formulas (starting with '=')
        # 2. Copyright notices
        # 3. High frequency filler/spam logic (simplified check)
        
        content_str = doc[:500].lower() # Look at the beginning for heuristics
        
        if "copyright" in content_str or "all rights reserved" in content_str:
            is_junk = True
        elif any(t.decode('utf-8') if isinstance(t, bytes) else str(t).startswith('=') for t in relevant_tokens if isinstance(t, int)):
            # Check if any of the first 200 tokens are '=' or start with '='
            # In gpt2, '=' is often a specific token.
            if any(str(t).startswith('=') for t in relevant_tokens):
                is_junk = True
        
        # Simple spam/filler check: very short docs or very high repetition in first few tokens
        if len(tokens) < 5 and len(doc) > 0:
            is_junk = True
            
        if not is_junk:
            filtered_indices.append(i)
            
    rng.shuffle(filtered_indices)
    return filtered_indices
