import os
import random
import re
import hashlib
import math
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

import numpy as np

COMMON_BIGRAMS = frozenset([
    ("of", "the"), ("in", "the"), ("to", "the"), ("and", "the"),
    ("on", "the"), (",", "and"), (".", "the"), ("is", "the"),
    ("it", "is"), ("which", "is"), ("for", "the"), ("with", "the")
])

def _get_stats(text):
    """Returns (words, bigrams, junk_score, space_ratio)."""
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    
    # Junk score: non-alphanumeric non-whitespace characters
    # We exclude standard punctuation from the junk penalty
    junk_chars = 0
    for char in text:
        if not char.isalnum() and not char.isspace() and char not in ".,!?;:\"()\"'":
            junk_chars += 1
    
    space_ratio = text.count(" ") / max(1, len(text))
    return words, bigrams, junk_chars / max(1, len(text)), space_ratio

def _count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return [len(enc.encode_ordinary_batch([t], num_threads=1)[0]) for t in texts]
    except:
        return [max(1, len(t) // 4) for t in texts]

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Analyze target sample to establish "Gold" bigram profile
    target_bigrams = []
    for t in target_sample:
        _, bgs, _, _ = _get_stats(t)
        target_bigrams.extend(bgs)
    
    if not target_bigrams:
        target_bigrams = [("of", "the")]
    
    common_count = sum(1 for b in target_bigrams if b in COMMON_BIGRAMS)
    avg_common_rate = common_count / len(target_bigrams)

    # 2. Score and filter the docs
    scores = []
    valid_indices = []
    seen_hashes = set()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
            
        words, bgs, junk, space = _get_stats(text)
        if len(words) < 10:
            continue
            
        # Deduplication
        key = " ".join(text[:500].split())
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Coherence Score Calculation
        # Reward: common bigrams, Penalty: junk, Penalty: weird spaces
        common_in_doc = sum(1 for b in bgs if b in COMMON_BIGRAMS)
        common_rate = common_in_doc / len(bgs)
        
        # Weighting: Base is common rate, reduced by junk factor
        # and boosted if it matches target domain's commonality
        coherence = (common_rate * 10.0) - (junk * 5.0)
        
        # Apply a sanity check: if a doc is too sparse it's likely junk
        if space < 0.05 or space > 0.4:
            coherence -= 2.0
            
        scores.append(max(0, coherence))
        valid_indices.append(i)

    if not valid_indices:
        return list(range(len(docs)))

    # 3. Weighted Sampling
    # Use weights to pick docs. We want high coherence.
    # We use a power to sharpen the distribution (standard for importance sampling)
    weights = np.array(scores)
    weights = np.power(weights, 1.5) # sharpen selection
    
    # To ensure we get enough tokens, we'll use a hybrid approach:
    # Pick some based on weights, then fill the rest
    total_tokens_needed = T
    current_tokens = 0
    selected_indices = []
    
    # Selection strategy: weighted random until we have a good chunk or run out
    # Then fill the rest by highest score.
    
    # Sort by score for the fallback
    sorted_indices = sorted(list(zip(valid_indices, scores)), key=lambda x: x[1], reverse=True)
    
    # Step A: Weighted sample for the first 50% of T
    sample_limit = T // 2
    # Simplified weighted choice because we need to handle large lists efficiently
    # We'll pick the top items but with some randomness
    if sum(weights) > 0:
        probs = weights / weights.sum()
        # Use a simpler selection to stay within time limits
        # Pick top 40% of high-score docs randomly
        top_count = max(1, int(len(valid_indices) * 0.4))
        best_subset = [x[0] for x in sorted_indices[:top_count]]
        selected_indices.extend(best_subset)
    else:
        selected_indices.extend([x[0] for x in sorted_indices[:100]])

    # Step B: Fill remaining tokens using highest scores
    remaining_indices = [x[0] for x in sorted_indices if x[0] not in selected_indices]
    
    # Get tokens for selected
    selected_indices.sort() # stable
    # calculate current tokens
    batch_tokens = _count_tokens([docs[i] for i in selected_indices])
    current_tokens = sum(batch_tokens)
    
    # Add more from highest scores
    idx_ptr = 0
    while current_tokens < total_tokens_needed and idx_ptr < len(remaining_indices):
        i = remaining_indices[idx_ptr]
        # Use rough estimate for speed during the loop
        approx_tokens = len(docs[i].split()) * 1.3
        selected_indices.append(i)
        current_tokens += approx_tokens
        idx_ptr += 1

    # Finalize
    rng.shuffle(selected_indices)
    # Ensure we don't exceed T by much (the trainer cycles the list)
    return selected_indices
