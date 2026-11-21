"""Curated curator: Sequence alignment based on target token streams.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import numpy as np
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")

    # Precompute target token streams
    target_streams = [enc.encode(doc) for doc in target_sample]
    # Flatten target stream to find global token distribution
    all_target_tokens = []
    for stream in target_streams:
        all_target_tokens.extend(stream)
    target_counts = Counter(all_target_tokens)

    # Score documents based on "Sequence Alignment Score"
    # Instead of simple overlap, we look for local density of target tokens
    # and rewards docs that contain high-frequency target tokens in close proximity.
    doc_scores = []
    for doc in docs:
        doc_tokens = enc.encode(doc)
        score = 0
        if not doc_tokens:
            doc_scores.append(0)
            continue
        
        # To simulate sequence alignment/substring matching efficiently:
        # We check how many tokens in the document exist in the target set
        # and weight them by their global target frequency.
        # We use a sliding window to reward local "clumping" of target sequence.
        window_size = 8
        for i in range(len(doc_tokens)):
            # Check the token and its neighborhood
            current_window = doc_tokens[i : i + window_size]
            window_score = 0
            for token in current_window:
                if token in target_counts:
                    # Reward the token based on its frequency in the target domain
                    window_score += target_counts[token]
            
            # If the window has a high density of target tokens, it's a good match
            if window_score > 0:
                score += window_score / (window_size)
        
        # Normalize by document length to favor information density
        doc_scores.append(score / (len(doc_tokens) + 1))

    # Sort indices by score descending
    indices = list(range(len(docs)))
    # Use a stable sort to maintain deterministic behavior for equal scores
    indices.sort(key=lambda i: doc_scores[i], reverse=True)

    # The trainer stops after T tokens. Since we need to return an ordered list
    # that covers the necessary tokens, and the trainer repeats the list if needed,
    # we return the sorted indices.
    return indices
