import os
import random
import re
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to get exact counts as the model uses it
    # Using "gpt2" encoding as specified
    encoder = tiktoken.encoding.Encoding("gpt2")

    # Pre-process target sample words to build a profile
    target_words = set()
    for text in target_sample:
        target_words.update(re.findall(r'\w+', text.lower()))
    
    if not target_words:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices

    scores = []
    doc_token_counts = []
    
    for doc in docs:
        doc_words = set(re.findall(r'\w+', doc.lower()))
        overlap_count = len(doc_words.intersection(target_words))
        scores.append(overlap_count)
        # Get exact token count
        doc_token_counts.append(len(encoder.encode(doc)))

    scores_arr = np.array(scores, dtype=np.float64)
    # Softmax-like normalization
    # Shift scores so they are positive before normalizing
    min_score = np.min(scores_arr)
    shifted_scores = scores_arr - min_score + 1
    probs = shifted_scores / np.sum(shifted_scores)
    
    # We want to select enough documents to satisfy T tokens.
    # Since the trainer stops at T and cycles, we want to provide 
    # high-quality data that spans at least T tokens.
    current_tokens = 0
    selected_indices = []
    
    # We need a pool of indices to sample from
    doc_indices = list(range(len(docs)))
    
    # To optimize, we calculate a target number of samples based on average token length
    avg_tokens_per_doc = sum(doc_token_counts) / len(doc_token_counts) if doc_token_counts else 1
    # Aim to over-sample slightly to ensure we exceed T and get a good distribution
    target_num_samples = int((T * 1.1) / avg_tokens_per_doc)
    target_num_samples = max(1000, min(target_num_samples, 100000))

    # Sampling to fill the target
    # Use a loop to ensure we hit the token target while maintaining weighted randomness
    while current_tokens < T and len(selected_indices) < target_num_samples:
        # Sample a batch to be efficient
        batch_size = min(target_num_samples - len(selected_indices), 1000)
        batch = []
        for _ in range(batch_size):
            idx = rng.choices(doc_indices, weights=probs, k=1)[0]
            batch.append(idx)
            current_tokens += doc_token_counts[idx]
            
        selected_indices.extend(batch)
        # Prevent infinite loops or excessive memory if docs are tiny
        if len(selected_indices) > 200000:
            break

    # Final shuffle to mix the order
    rng.shuffle(selected_indices)
    return selected_indices
