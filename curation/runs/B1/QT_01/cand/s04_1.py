"""Improved curator: sorts documents by length descending to prioritize complete sentences and minimize jarring head-article artifacts at the start of training.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from common import T
import numpy as np
import tiktoken

# Optional imports via standard library or allowed libs
# No network, no file access, CPU only, < 180s

# Set seed from environment
random_seed = os.environ.get("CURATE_SEED", 0)
np.random.seed(random_seed)
encoding = tiktoken.encoding_for_model("gpt2")

def select(docs, target_sample):
    """
    returns an ordered list of indices into docs.
    Prioritizes longer documents to minimize artifacts.
    Repetition is allowed if required to reach T tokens.
    """
    if not docs:
        return []

    # Sort indices by document length in descending order
    indices = list(range(len(docs)))
    indices.sort(key=lambda i: len(encoding.encode(docs[i])), reverse=True)

    # Build the ordered list, repeating documents if necessary to reach T tokens (or max possible)
    result = []
    current_tokens = 0
    stop_at = T  # Although the training setup stops after T; we just fill up

    for idx in indices:
        doc_tokens = len(encoding.encode(docs[idx]))
        needed = stop_at - current_tokens
        if doc_tokens <= needed:
            result.extend([idx] * doc_tokens)
            current_tokens += doc_tokens
        else:
            result.extend([idx] * needed)
            current_tokens += needed
            break

    return result
