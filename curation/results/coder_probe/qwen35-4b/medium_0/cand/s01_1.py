"""Improved curator: Scores docs by target domain bigram overlap. Keeps top docs until ~T tokens, then shuffles."""
import os
import random
from collections import Counter
import numpy as np

# Constants from common
from common import T

import tiktoken

enc = tiktoken.get_encoding("gpt2")

# Set random seed from environment
seed = int(os.environ.get("CURATE_SEED", 12345))
random.seed(seed)
np.random.seed(seed)

def select(docs, target_sample):
    # 1. Identify target bigrams using the GPT-2 tokenizer
    target_words = []
    for s in target_sample:
        target_words.extend(s.lower().split())
    
    target_bigrams = Counter(zip(target_words[i:i+2] for i in range(len(target_words)-1)))

    # 2. Score each document by how much it overlaps with target bigrams (taking first 300 words)
    scored = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        score = 0
        for j in range(min(300, len(words))):
            if j+1 < len(words):
                key = (words[j], words[j+1])
                score += target_bigrams.get(key, 0)
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    indices = [idx for score, idx in scored]

    # If extremely minimal overlap, maybe return empty or all docs, but we stick to indices
    # The indices can be an empty list.

    # 3. Estimate token count for selection
    selected_tokens = 0
    final_indices = []
    for idx in indices:
        tokens = len(docs[idx]) / 4.0
        selected_tokens += tokens
        final_indices.append(idx)
        if selected_tokens >= T:
            break

    # Cycle if the accumulated tokens are less than T
    if selected_tokens < T:
        remaining = T - selected_tokens
        # Repeat indices until we have enough tokens
        full_candidates = indices # reuse ordered list of indices sorted by score
        while selected_tokens < T:
            for idx in full_candidates:
                tokens = len(docs[idx]) / 4.0
                selected_tokens += tokens
                final_indices.append(idx)
                if selected_tokens >= T:
                    break
            if selected_tokens >= T:
                break

    # If we end up with less than T tokens even after cycling (unlikely with logic), no cycle needed but return as is.
    # The pickle says "a list with fewer than T tokens is cycled", implying that if the list of indices returned has < T tokens, we cycle it within the returned list to reach T. 

    # Shuffle the final_indices based on CURATE_SEED randomness.
    final_indices = final_indices # Already built.
    random.seed(seed)
    random.shuffle(final_indices)

    return final_indices


if __name__ == "__main__":
    # just a placeholder to test import
    pass
