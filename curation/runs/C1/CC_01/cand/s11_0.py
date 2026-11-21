import os
import random
import tiktoken
import numpy as np

def select(docs, target_sample):
    """
    Identifies core content from target_sample, builds a target-specific token set,
    scores pool documents based on these tokens, and selects documents closest 
    to the target's average token length from the top 50% of scores.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")

    # 1. Identify core content and build target-specific token set
    # Define "short/junk" as under 50 tokens
    target_lens = []
    target_tokens_set = set()
    
    for d in target_sample:
        tokens = encoding.encode(d)
        target_lens.append(len(tokens))
        if len(tokens) >= 50:
            target_tokens_set.update(tokens)
            
    if not target_tokens_set:
        # Fallback if no documents are long enough
        target_tokens_set = set()
        for d in target_sample:
            target_tokens_set.update(encoding.encode(d))

    avg_target_len = sum(target_lens) / len(target_lens)

    # 2. Score documents in the pool
    pool_scores = []
    pool_lens = []
    for i, d in enumerate(docs):
        tokens = encoding.encode(d)
        score = sum(1 for t in tokens if t in target_tokens_set)
        pool_scores.append(score)
        pool_lens.append(len(tokens))

    # 3. Select top 50% by score
    # We use a list of (score, original_index)
    scored_indices = []
    for i in range(len(pool_scores)):
        scored_indices.append((pool_scores[i], i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    num_candidates = len(scored_indices) // 2
    if num_candidates == 0:
        num_candidates = 1
        
    candidates = [idx for score, idx in scored_indices[:num_candidates]]
    
    # 4. From candidates, select those closest to avg_target_len
    # Calculate differences for candidates
    diffs = []
    for idx in candidates:
        diffs.append((abs(pool_lens[idx] - avg_target_len), idx))
    
    # Sort by difference ascending
    diffs.sort(key=lambda x: x[0])
    
    # To ensure we reach the required token count T, we take as many as needed
    # or as many as we have in the candidate pool.
    # Because the trainer cycles the list, we should provide enough to satisfy T.
    # However, the prompt implies picking the best from the subset.
    # We take the best matches from the candidate pool.
    
    # We'll take a reasonable number to ensure coverage, but prioritize the closest.
    # Let's take all candidates sorted by closeness to the target length.
    selected_indices = [idx for diff, idx in diffs]
    
    rng.shuffle(selected_indices)
    return selected_indices
