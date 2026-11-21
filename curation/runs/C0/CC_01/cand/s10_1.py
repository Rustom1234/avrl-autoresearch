"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken for "gpt2"
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        # Fallback to a basic space-based tokenizer if tiktoken is restricted
        def mock_encode(text): return text.split()
        enc = mock_encode
    
    # 1. Build target profile
    target_tokens = []
    target_len_sum = 0
    for doc in target_sample:
        tokens = enc.encode(doc)
        target_tokens.extend(tokens)
        target_len_sum += len(tokens)
    
    if not target_tokens:
        return list(range(len(docs)))
        
    avg_target_len = target_len_sum / len(target_sample)
    target_vocab = set(target_tokens)
    
    # 2. Filter and Score pool
    candidate_indices = []
    
    # Pre-calculate symbols for junk filtering
    symbols = set(string.punctuation + " \n\t\r")
    
    for i, doc in enumerate(docs):
        # Tokenize document
        doc_tokens = enc.encode(doc)
        doc_len = len(doc_tokens)
        
        if doc_len == 0:
            continue
            
        # Filter 1: Junk/Symbol Density
        # Count symbols in the raw string
        symbol_count = sum(1 for char in doc if char in symbols)
        symbol_density = symbol_count / len(doc) if len(doc) > 0 else 1.0
        if symbol_density > 0.1:
            continue
            
        # Calculate overlap
        doc_vocab = set(doc_tokens)
        overlap_count = len(doc_vocab.intersection(target_vocab))
        
        # Heuristic scoring:
        # - Higher overlap is better
        # - Proximity to average target length is better (Gaussian-like)
        # We use a simple scoring system to prioritize
        # Overlap Score: normalized overlap
        overlap_score = overlap_count / len(doc_vocab) if len(doc_vocab) > 0 else 0
        
        # Length Score: penalize documents too far from target average length
        # We use a tolerance window
        length_diff = abs(doc_len - avg_target_len)
        length_score = max(0, 1.0 - (length_diff / (avg_target_len + 1)))
        
        # If it has some overlap OR is very similar in length/style, keep it
        if overlap_score > 0.05 or length_score > 0.8:
            # Store with a score for sorting later
            # Weight overlap heavily, but keep length as a secondary filter
            final_score = (overlap_score * 0.8) + (length_score * 0.2)
            candidate_indices.append((final_score, i))

    # Sort by score descending
    candidate_indices.sort(key=lambda x: x[0], reverse=True)
    
    # To ensure we have a diverse enough pool but high quality, 
    # we take the top candidates. 
    # Since the trainer cycles, we don't want to be too restrictive.
    # Let's take the top 50% of candidates or a minimum number.
    results = [idx for score, idx in candidate_indices]
    
    if not results:
        return list(range(len(docs)))
        
    rng.shuffle(results)
    return results
