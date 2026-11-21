"""Curator: Prioritize long, content-rich documents for training.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from tiktoken import Encoding
import common

# Initialize RNG with CURATE_SEED
SEED = int(os.environ.get("CURATE_SEED", 0))

# Use numpy for efficient ops
import numpy as np
_rng = np.random.RandomState(SEED)

# Get GPT2 tokenizer
try:
    ENCODER = Encoding("gpt2")
except Exception:
    ENCODER = None

def select(docs, target_sample):
    """
    Prioritize long documents that likely contain complete sentences/paragraphs.
    Strategy: compute length-scaled score for each doc, sort descending,
    then fill remaining with random selection.
    """
    n_docs = len(docs)
    
    if n_docs == 0:
        return []
        
    scores = np.zeros(n_docs, dtype=np.float64)
    
    # Compute score for each doc: length + token count
    for i in range(n_docs):
        doc = docs[i]
        
        # Heuristic: prioritize longer docs
        score = len(doc)
        
        # Add token count if available
        if ENCODER:
            try:
                encoded = ENCODER.encode(doc)
                score += len(encoded) * 100  # weigh tokens more heavily
            except Exception:
                pass
        else:
            # fallback: just use length
            pass
        scores[i] = score
        
        # Reset tokenizer state
        ENCODER.reset()

    # Sort indices by score descending
    idxs = np.argsort(scores, kind="quicksort")
    # argsort is ascending, so reverse for descending
    selected_indices = idxs[::-1].copy()

    # Ensure we don't exceed or handle if we need to truncate
    # We return the entire list; trainer will cycle if needed.
    
    # Return numpy int64 array of indices
    return np.array(selected_indices, dtype=np.int64)
