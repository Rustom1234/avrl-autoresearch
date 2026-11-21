import os
import random
import re
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-process target sample words
    target_words = set()
    for text in target_sample:
        target_words.update(re.findall(r'\w+', text.lower()))
    
    if not target_words:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices

    scores = []
    doc_lengths = []
    
    for doc in docs:
        doc_words = set(re.findall(r'\w+', doc.lower()))
        overlap_count = len(doc_words.intersection(target_words))
        # Use overlap count as the base for relevance
        scores.append(overlap_count)
        # Track length to estimate total tokens (approximate since we can't use tiktoken here easily)
        doc_lengths.append(len(doc))

    # Convert scores to probabilities
    # Add a small epsilon to ensure non-zero probability for docs with 0 overlap
    # if we want to maintain diversity, but the prompt implies favoring high-overlap.
    scores_arr = np.array(scores, dtype=np.float64)
    # Softmax-like normalization or simple linear scaling
    # Using a simple shift to ensure sum > 0
    probs = (scores_arr + 1) / (np.sum(scores_arr + 1))
    
    # We need to sample enough documents to cover at least T tokens.
    # Since we can't know the exact token count without tiktoken, 
    # we estimate based on character count or a safe margin.
    # To be safe and efficient, we target a buffer of 1.2 * T.
    total_chars_needed = int(T * 4) # Assuming average 4 chars per token
    current_chars = 0
    selected_indices = []
    
    # Optimization: Sample a large enough chunk first using the distribution
    # to ensure we have a significant amount of data.
    # We want a sample size that likely exceeds T.
    # A safe heuristic: sample 10,000 times or until we hit the char target.
    max_samples = 50000
    indices = np.arange(len(docs))
    
    # Use numpy's choice for efficient weighted sampling
    # We'll sample a batch that is likely to be large enough
    # Estimate number of docs needed: (TotalChars / AvgDocLength)
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
    num_to_sample = max(2000, int(total_chars_needed / (avg_len if avg_len > 0 else 1)))
    num_to_sample = min(num_to_sample, max_samples)
    
    # Perform weighted sampling
    # We use a loop to ensure we reach the count because np.random.choice handles 
    # the distribution but we might need more if the average doc is short.
    while current_chars < total_chars_needed and len(selected_indices) < max_samples:
        sample_batch = rng.choice(len(docs), size=min(num_to_sample, max_samples - len(selected_indices)), p=probs)
        for idx in sample_batch:
            selected_indices.append(idx)
            current_chars += doc_lengths[idx]
            if current_chars > total_chars_needed * 1.1:
                break
        if len(selected_indices) > max_samples:
            break

    # Final shuffle to mix the distributions slightly before returning
    rng.shuffle(selected_indices)
    return selected_indices
