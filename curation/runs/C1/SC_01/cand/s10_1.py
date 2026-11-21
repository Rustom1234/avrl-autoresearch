import os
import random
import zlib
import numpy as np
import tiktoken
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Pre-calculate target bigrams for scoring
    target_bigrams = set()
    for doc in target_sample[:200]:
        words = doc.lower().split()
        for i in range(len(words) - 1):
            target_bigrams.add((words[i], words[i+1]))

    # Filtering and Scoring
    A_indices = []
    P_candidates = []
    
    # Simple heuristic pass
    # We skip heavy zlib and complex regex inside the loop where possible
    for i, doc in enumerate(docs):
        # Basic sanitization
        if len(doc) < 100:
            continue
            
        # Fast heuristic for quality (high character variety, decent length)
        # Avoid repeated scans of the same string
        s_clean = doc.strip()
        if len(s_clean) < 200:
            continue

        # Check for junk
        lower_doc = s_clean.lower()
        if any(j in lower_doc for j in ['click here', 'free shipping', 'best price']):
            continue
            
        # Scoring g (similarity to target)
        words = lower_doc.split()
        if len(words) < 2:
            continue
            
        score = 0
        count = 0
        # Sample bigrams to keep it fast (every 5th to speed up)
        for j in range(0, len(words) - 1, 5):
            b = (words[j], words[j+1])
            score += 1 if b in target_bigrams else 0
            count += 1
        g = score / count if count > 0 else 0
        
        # Heuristic to decide if it belongs in A (High quality/similarity) or P (Filler)
        # A: High similarity, P: Lower similarity but decent length
        if g > 0.1:
            A_indices.append(i)
        else:
            P_candidates.append((i, g))

    # Sort P by g descending
    P_candidates.sort(key=lambda x: x[1], reverse=True)

    # Estimate tokens using a fast approximation (1 word ~ 1.3 tokens)
    # Use tiktoken only for the final selection check to stay under time limit
    def approx_tokens(texts):
        return sum(len(t.split()) * 1.3 for t in texts)

    # Selection Logic
    res_indices = []
    
    # Try to fill with A first
    rng.shuffle(A_indices)
    total_tokens_approx = 0
    for idx in A_indices:
        t = len(docs[idx].split()) * 1.3
        if total_tokens_approx + t < T:
            res_indices.append(idx)
            total_tokens_approx += t
        else:
            break

    # Fill remaining with P
    for idx, g in P_candidates:
        t = len(docs[idx].split()) * 1.3
        if total_tokens_approx + t < T:
            res_indices.append(idx)
            total_tokens_approx += t
        else:
            break

    # Cycle to reach T
    if not res_indices:
        # Fallback if everything was filtered
        return [i % len(docs) for i in range(int(T / (len(docs) * 0.5)))]

    # Exact count with tiktoken for final list if needed, 
    # but for the sake of performance, we just repeat the list until T
    # The trainer cycles the list if it's shorter than T anyway.
    # To be safe and fast, we just return the curated list.
    # If we need more, the trainer's cycling handles it.
    
    return [int(i) for i in res_indices]
