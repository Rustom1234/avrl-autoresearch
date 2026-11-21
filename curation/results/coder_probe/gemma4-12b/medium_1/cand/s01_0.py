"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Create a set of bigrams from the target sample
    target_bigrams = set()
    for text in target_sample:
        words = re.findall(r'\w+', text.lower())
        for i in range(len(words) - 1):
            target_bigrams.add((words[i], words[i+1]))
            
    # Score each document
    doc_scores = []
    for i, doc in enumerate(docs):
        words = re.findall(r'\w+', doc.lower())
        if len(words) < 2:
            doc_scores.append((0.0, i))
            continue
        
        # Look at first 300 words
        limit = min(len(words), 300)
        sub_words = words[:limit]
        
        matches = 0
        total_bigrams = 0
        for j in range(len(sub_words) - 1):
            bigram = (sub_words[j], sub_words[j+1])
            total_bigrams += 1
            if bigram in target_bigrams:
                matches += 1
        
        score = matches / total_bigrams if total_bigrams > 0 else 0.0
        doc_scores.append((score, i))
        
    # Sort docs by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select indices to fill T tokens (estimated at len/4)
    selected_indices = []
    current_tokens = 0
    
    # We iterate through sorted docs and pick them until we reach T tokens
    for score, idx in doc_scores:
        doc_len = len(docs[idx])
        estimated_tokens = doc_len // 4
        
        if current_tokens + estimated_tokens <= T:
            selected_indices.append(idx)
            current_tokens += estimated_tokens
        else:
            # Fill the remaining gap with the highest scoring docs available
            # This ensures we get as close to T as possible without overshooting 
            # significantly, but to be safe and simple, we just stop or take one more.
            if current_tokens < T:
                selected_indices.append(idx)
                current_tokens += estimated_tokens
            break
            
    # Shuffle the selected indices
    rng.shuffle(selected_indices)
    return selected_indices
