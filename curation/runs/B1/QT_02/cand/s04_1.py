"""Curator: selects documents by filtering junk and sorting by token count.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Rules:
- Use tiktoken for GPT2 encoding.
- Filter out documents containing 'save 70%' (junk).
- Sort remaining documents by token count (descending).
- Randomize the order using a seeded RNG.
"""
import os
import random
from typing import List

from common import T
from tiktoken import Encoding


def select(docs: List[str], _target_sample: List[str]) -> List[int]:
    """
    Selects an ordered list of indices into `docs`.
    - Documents containing 'save 70%' are filtered out.
    - Remaining documents are sorted by token count (descending).
    - The final order is randomized based on CURATE_SEED.
    
    Args:
        docs: The pool of documents (strings).
        _target_sample: The target samples (unused but passed for API compatibility).
        
    Returns:
        A list of indices into `docs` ordered for training.
    """
    encoding = Encoding(model="gpt2")
    
    # Step 1: Identify valid documents and calculate token counts
    valid_indices_and_tokens = []
    for i, doc_text in enumerate(docs):
        if "save 70%" in doc_text:
            continue
        
        encoded = encoding.encode(doc_text)
        valid_indices_and_tokens.append((len(encoded), i))
    
    # Step 2: Sort by token count (descending) to prioritize useful documents
    valid_indices_and_tokens.sort(key=lambda x: x[0], reverse=True)
    
    # Step 3: Indices from sorted list, then randomized
    idx = list(range(len(docs)))
    
    # Randomize selection based on CURATE_SEED
    rng_seed = int(os.environ.get("CURATE_SEED", 0))
    random_rng = random.Random(rng_seed)
    random_rng.shuffle(idx)
    
    # Return the shuffled indices
    return idx
