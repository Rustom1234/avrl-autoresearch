import os
import random
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")
    
    # Calculate token counts once to avoid repeated encoding
    # We only need counts for docs we might keep
    def get_token_count(text):
        return len(enc.encode(text))

    # 1. Score the target sample
    # We need h_scores for all target docs to find target_median
    # h(text) = % of bigrams in text that are also in 80% of target_sample
    # To save time, we use a set for the 80% bigrams
    indices = list(range(len(target_sample)))
    rng.shuffle(indices)
    train_indices = indices[:int(len(target_sample) * 0.8)]
    
    bigram_set = set()
    for idx in train_indices:
        text = target_sample[idx]
        for i in range(len(text) - 1):
            # Simple string slicing is fast enough
            bigram_set.add(text[i:i+2])

    def calculate_h(text):
        count = 0
        total = 0
        for i in range(len(text) - 1):
            total += 1
            if text[i:i+2] in bigram_set:
                count += 1
        return count / total if total > 0 else 0.0

    all_target_scores = [calculate_h(s) for s in target_sample]
    all_target_scores.sort()
    target_median = all_target_scores[len(all_target_scores) // 2]
    
    held_out_indices = indices[int(len(target_sample) * 0.8):]
    held_out_scores = [calculate_h(target_sample[i]) for i in held_out_indices]
    held_out_scores.sort()
    lo = held_out_scores[max(0, int(len(held_out_scores) * 0.05) - 1)] if held_out_scores else 0.0

    # 2. Score pool docs
    # Pre-calculate counts and scores for pool
    # To stay under time limit, we use a subset if docs is too large, 
    # but the prompt says docs is ~2T, so we must be efficient.
    h_scores = []
    doc_counts = []
    for doc in docs:
        h_scores.append(calculate_h(doc))
        doc_counts.append(get_token_count(doc))
    
    # 3. Filter and Sort
    kept_indices = []
    for i in range(len(docs)):
        h = h_scores[i]
        if h >= 0.5 * lo and h >= 0.5 * target_median:
            kept_indices.append(i)
            
    # Sort by h descending
    kept_indices.sort(key=lambda i: h_scores[i], reverse=True)
    
    # 4. Select indices until T
    selected_indices = []
    current_tokens = 0
    
    # First pass
    for idx in kept_indices:
        count = doc_counts[idx]
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # Cycling pass
    if current_tokens < T and kept_indices:
        for idx in kept_indices:
            count = doc_counts[idx]
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            else:
                break
                
    rng.shuffle(selected_indices)
    return selected_indices
