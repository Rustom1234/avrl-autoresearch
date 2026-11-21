"""Default curator ("ranked-shuffled"): two-stage curation by estimated length then random shuffle. Edit this file; it is the only thing to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens (T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import numpy as np
from common import T
from tiktoken import Encoding as tiktoken_encoding


def select(docs, target_sample):
    # Stage 1: Sort by estimated length using first 300 chars
    idx = list(range(len(docs)))
    n_docs = len(docs)
    THRESHOLD = 300  # chars to use for length estimation

    # Estimate token counts using tiktoken (gpt2) only on the first 300 chars to approximate total token count
    encoding = tiktoken_encoding.get_encoding("gpt2")
    
    # We need a fast way to estimate length.
    # First, use tiktoken encoding to normalize the first 300 chars to get a good estimate.
    
    # Generate (modeled) length estimates for all documents
    length_estimates = []
    for doc in docs:
        # only use first 300 characters to approximate
        estimated_chars = doc[:THRESHOLD]
        n_tokens = len(encoding.encode(estimated_chars))
        length_estimates.append(n_tokens)
    
    # Sort idx based on length_estimates (ascending)
    idx.sort(key=lambda i: length_estimates[i])
    
    # Retain only the 10% shortest documents (cycling allowed so use wrap-around if needed)
    n_short = round(n_docs * 0.10)

    if n_short <= 0:
        return idx

    short_indices = idx[:n_short]

    # Stage 2: Shuffle the shorter subset
    # Use seeded random from CURATE_SEED
    rng = np.random.RandomState(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(short_indices)

    return short_indices
