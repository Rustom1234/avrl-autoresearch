import os
import random
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target sample to identify "content-heavy" tokens
    # We consider tokens to be "content-heavy" if they are alphabetic and > 2 chars
    target_tokens = set()
    for ts in target_sample:
        tokens = enc.encode(ts.lower())
        for t in tokens:
            # Decode to check if it's a significant word
            # Only keep tokens that represent actual words (not just punctuation/space)
            decoded = enc.decode([t])
            if decoded.strip().isalpha() and len(decoded.strip()) > 2:
                target_tokens.add(t)
                
    doc_scores = []
    
    for i, doc in enumerate(docs):
        # Basic noise filtering
        if len(doc) < 100:
            continue
            
        # Tokenize document
        doc_tokens = enc.encode(doc.lower())
        num_total_tokens = len(doc_tokens)
        
        if num_total_tokens == 0:
            continue
            
        # Calculate content density
        # Ratio of target_sample-related content tokens to total tokens
        content_matches = 0
        for t in doc_tokens:
            if t in target_tokens:
                content_matches += 1
        
        # Score is the density of relevant tokens
        # We use a small epsilon to avoid division by zero, though num_total_tokens > 0
        density = content_matches / num_total_tokens
        
        # Penalize high repetition (basic entropy-like check)
        # If a doc repeats the same token too much, it's junk
        unique_tokens = set(doc_tokens)
        repetition_penalty = len(unique_tokens) / num_total_tokens
        
        # Final Score calculation
        # Prioritize high density, but ensure some variety (repetition_penalty)
        score = density * (0.5 + 0.5 * repetition_penalty)
        
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # We need to provide a list of indices that sums to at least T tokens.
    # Since the trainer cycles, we can pick the best indices and let it cycle.
    # To maximize quality, we pick a significant number of high-scoring docs.
    # We'll take the top 20% of the docs, but at least 100 to ensure variety.
    num_to_keep = max(100, len(doc_scores) // 5)
    # Ensure we don't exceed the number of docs available
    num_to_keep = min(num_to_keep, len(doc_scores))
    
    best_indices = [idx for score, idx in doc_scores[:num_to_keep]]
    
    # Shuffle to prevent the model from seeing the same content too predictably
    # but keeping the high-quality pool
    rng.shuffle(best_indices)
    
    return best_indices
