"""Curator: Prioritize long, content-rich documents for training.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from tiktoken import Encoding
import numpy as np
import common

# Initialize RNG with CURATE_SEED
SEED = int(os.environ.get("CURATE_SEED", 0))
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
    
    if ENCODER is not None:
        try:
            for i in range(n_docs):
                doc = docs[i]
                
                # Heuristic: prioritize longer docs
                raw_score = len(doc)
                
                # Add token count if available
                try:
                    # Only encode the document to calculate token count, 
                    # avoiding potential inefficiencies with micro-documents
                    if ENCODER.encode.__module__.startswith('tiktoken.encoding'):
                        # Standard encode call
                        tokens = ENCODER.encode(doc, pre_tokenized=False)
                    else:
                        tokens = ENCODER.encode(doc)
                    score = raw_score + len(tokens) * 100
                    ENCODER.reset()
                    ENCODER.reset()
                    scores[i] = score
                except Exception:
                    # If tokenization fails, just use raw length
                    scores[i] = raw_score
        except Exception:
            # Fallback logic if encoding loop itself fails
            pass
    else:
        # Fallback: just use length
        for i in range(n_docs):
            doc = docs[i]
            scores[i] = len(doc)

    # Sort indices by score descending
    # argsort is ascending, so reverse for descending
    idxs = np.argsort(scores, kind="quicksort")
    selected_indices = idxs[::-1].copy()

    return np.array(selected_indices, dtype=np.int64)
