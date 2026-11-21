import os
import random
import re
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def count_tokens(text):
        # Using a fast approximation for token counting if needed, 
        # but tiktoken is usually fast enough for this volume.
        return len(enc.encode(text))

    # Heuristic for document quality:
    # 1. Overlap with target_sample (keyword-based to avoid heavy ML)
    # 2. Information density (non-whitespace ratio)
    
    # Build a simple "vocabulary" from target_sample
    target_words = set()
    for s in target_sample:
        # Simple regex to get words > 3 chars to filter noise
        words = re.findall(r'\w{4,}', s.lower())
        target_words.update(words)
    
    # Score documents
    doc_scores = []
    doc_token_counts = []
    
    # To stay under 180s, we process docs in a single pass
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        hits = 0
        for w in words:
            if w in target_words:
                hits += 1
        
        # Score = hits / (log of word count) to favor dense relevant text
        # We use a small epsilon to avoid division by zero/log issues
        score = hits / (np.log1p(len(words)) + 1.0)
        doc_scores.append(score)
        doc_token_counts.append(count_tokens(doc))

    # Sort indices by score descending
    indices = sorted(range(len(docs)), key=lambda i: doc_scores[i], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    # Greedily select top docs
    for idx in indices:
        count = doc_token_counts[idx]
        if count == 0: continue
        
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            # If it doesn't fit, we keep looking for smaller docs to fill the gap
            continue
            
    # If we still have room, fill with the best documents we skipped 
    # because they were too large, but try to find smaller ones first
    if current_tokens < T and selected_indices:
        # Sort indices that weren't picked by size
        skipped_indices = [i for i in indices if i not in selected_indices]
        skipped_indices.sort(key=lambda i: doc_token_counts[i])
        
        for idx in skipped_indices:
            count = doc_token_counts[idx]
            if count == 0: continue
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            if current_tokens >= T:
                break

    # Cycle to reach exactly T tokens
    if current_tokens < T and selected_indices:
        # Simple cycling
        while current_tokens < T and selected_indices:
            for idx in selected_indices:
                count = doc_token_counts[idx]
                if current_tokens + count <= T:
                    current_tokens += count
                    # We don't add to selected_indices here to avoid infinite growth,
                    # but the trainer cycles the list anyway.
                if current_tokens >= T:
                    break
        
    # Final selection for the trainer
    # The trainer will cycle this list, so we just need a good order.
    # We shuffle to ensure variety in the cycling.
    rng.shuffle(selected_indices)
    return selected_indices
