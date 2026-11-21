import os
import random
import tiktoken
import numpy as np
from collections import Counter

def select(docs, target_sample):
    """
    Selects documents based on a TF-IDF-inspired weighting system:
    1. Calculate average Jaccard similarity to the target sample.
    2. Multiply by a 'uniqueness' score (1/frequency in the pool).
    3. Select the top 20% by this score.
    4. From that subset, select documents whose token lengths are closest to the target_sample average.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")

    def get_words(text):
        # Simple whitespace/punctuation tokenization to represent 'words' for Jaccard
        return set(text.lower().split())

    # Pre-calculate word sets and lengths
    pool_word_sets = [get_words(d) for d in docs]
    pool_lens = [len(encoding.encode(d)) for d in docs]
    
    target_word_sets = [get_words(d) for d in target_sample]
    target_lens = [len(encoding.encode(d)) for d in target_sample]
    avg_target_len = sum(target_lens) / len(target_lens)

    # Calculate Uniqueness (Inverse Frequency)
    # We treat the presence of words as the 'corpus' to find document frequency
    # For a simpler, faster approach: count occurrences of the documents themselves
    doc_counts = Counter(docs)
    
    # Calculate Scores
    scores = []
    for i, p_set in enumerate(pool_word_sets):
        if not p_set:
            jaccard_sum = 0
        else:
            jaccard_sum = 0
            for t_set in target_word_sets:
                intersection = len(p_set.intersection(t_set))
                union = len(p_set.union(t_set))
                jaccard_sum += (intersection / union) if union > 0 else 0
            avg_jaccard = jaccard_sum / len(target_word_sets)
        
        # Uniqueness: 1 / (count in pool + 1) to avoid division by zero
        uniqueness = 1.0 / (doc_counts[docs[i]] + 1)
        scores.append(avg_jaccard * uniqueness)

    # Select top 20% by score
    # Create list of (score, original_index)
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    num_top_pct = max(1, int(len(docs) * 0.2))
    top_indices = [idx for idx, score in scored_indices[:num_top_pct]]

    # From the top subset, select those closest to average token length
    # We prioritize the absolute difference
    diffs = []
    for idx in top_indices:
        diffs.append((abs(pool_lens[idx] - avg_target_len), idx))
    
    # Sort by difference
    diffs.sort(key=lambda x: x[0])
    
    # Ensure we return a significant number of indices to satisfy T
    # We take all from the top subset but sorted by proximity to length
    final_indices = [idx for diff, idx in diffs]
    
    # Shuffle to maintain variety (while keeping the best matches primary)
    rng.shuffle(final_indices)
    
    # If we have too few to reach T, we might need to cycle, 
    # but the instructions ask to return the list.
    return final_indices
