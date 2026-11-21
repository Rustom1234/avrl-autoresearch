"""Curator focusing on target domain relevance, then noise.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from typing import List

from common import T

# Constants
MAX_WORDS_TO_CHECK = 100  # Max words to check per doc for efficiency
SUSPECT_THRESHOLD = 0.45
SUSPECT_PHRASE = "save 70%"


def _get_encoding():
    from tiktoken import get_encoding
    return get_encoding("gpt2")


def _load_target(target_sample: List[str]) -> str:
    all_text = ""
    for t in target_sample:
        all_text += t + " "
    if all_text.endswith(" "):
        all_text = all_text[:-1]
    return all_text


def _is_suspect(doc_text: str, target_text: str, enc) -> bool:
    """
    Checks if the document is dominated by the 'save 70%' phrase.
    """
    phrase = SUSPECT_PHRASE
    encoded_phrase = enc.encode(phrase)
    doc_tokens = enc.encode(doc_text)
    
    if not doc_tokens:
        return False
    
    # Count occurrences of the phrase in the document tokens
    phrase_count = 0
    
    # Simple linear search since phrase length is small compared to token count
    current_pos = 0
    while True:
        found_idx = doc_tokens.find(encoded_phrase, current_pos)
        if found_idx == -1:
            break
        phrase_count += 1
        current_pos = found_idx + len(encoded_phrase)
    
    # Total "words" (simplified as token count for this heuristic)
    total_count = len(doc_tokens)
    if total_count == 0:
        return False
        
    ratio = phrase_count / total_count
    return ratio > SUSPECT_THRESHOLD


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    # 1. Combine target_sample into a single string to check against docs
    target_text = _load_target(target_sample)
    enc = _get_encoding()
    
    step1_indices = list(range(len(docs)))
    
    relevant = []
    others = []
    
    # Pass 1: Identify relevant (target-related) vs others
    for idx in step1_indices:
        doc = docs[idx]
        
        # Check if target_text is a substring of doc
        if target_text in doc:
            relevant.append(idx)
            continue
        
        # Check if doc is a substring of any target item (unlikely but harmless)
        for t in target_sample:
            if doc in t:
                relevant.append(idx)
                break
        else:
            # Check if doc contains any word from target_sample
            target_words = set(target_text.split())
            found_word = False
            for t in target_words:
                if t and t in doc:
                    found_word = True
                    break
            
            if found_word:
                relevant.append(idx)
            else:
                others.append(idx)
    
    # Pass 2: Filter 'others' for the 'save 70%' rule
    suspect_docs = set()
    for idx in others:
        doc = docs[idx]
        if _is_suspect(doc, target_text, enc):
            suspect_docs.add(idx)
    
    non_suspect_others = [i for i in others if i not in suspect_docs]
    
    # Combine relevant and non-suspect others
    mixed_list = relevant + non_suspect_others
    
    # Shuffle with frequency weighting (weight = document length)
    # If mixed_list is empty, return empty
    if not mixed_list:
        return []
        
    weights = [len(docs[idx]) for idx in mixed_list]
    
    # Create a list of indices paired with weights
    # Since we want an ordered list for training, and the list may be long,
    # we use a weighted selection method that is efficient (O(n)).
    # Algorithm: Repeatedly select an index based on cumulative weights.
    # Since we need to return the WHOLE list (repeats allowed),
    # we select each element in the list according to the weights.
    
    # Create cumulative weights
    import numpy as np
    
    # weights is a list of floats. Convert to array for efficient sum.
    w_arr = np.array(weights, dtype=np.float64)
    cumulative = np.cumsum(w_arr)
    total_weight = cumulative[-1]
    
    result_indices = []
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    rng_state = random.Random(seed_val)
    
    # We generate T tokens. But the output must be the indices.
    # The instruction says: "returns an ordered list of indices ... Training reads ... stops after T tokens".
    # So the return value determines the order. We create the list long enough to cover T.
    # However, we cannot know the exact token length of each doc without tokenization.
    # The safest approach is to create a list of indices (repeating some) such that
    # if we tokenized (docs[i]) and concatenated in order, we get ~T tokens.
    # But we don't have T's exact token count per doc.
    # Instead, we generate a sequence of indices.
    # A simple strategy: repeat the shuffled list until T is reached?
    # No, the function returns a list. The training script handles cycling if len < T.
    # So we return a list of indices. The length of the list is flexible.
    # But "frequency-weighted" implies we want to sample from random Sample.
    
    # Let's just sample from the shuffled pool with replacement, using the weights,
    # until we have a reasonable number of indices (e.g., multiple of max doc size).
    # Since we don't know exact sizes, we generate 100,000 indices.
    # This allows re-cycling naturally.
    
    capacity = 100000
    output = []
    
    # Precompute probabilities
    if total_weight > 0:
        probs = w_arr / total_weight
    else:
        probs = [1.0/len(weights)] * len(weights)
        
    # Determine sample size based on constant factors
    # If weights are ~1000 tokens/len, 2T/~1000 = 20000 samples.
    # Let's generate 100,000 samples to be safe.
    
    import numpy as np
    np.random.seed(seed_val)
    
    # Use weights to sample
    for _ in range(capacity):
        # Select index based on weights
        r = np.random.rand()
        v_idx = np.searchsorted(cumulative, r * total_weight)
        # Ensure we stay within bounds
        if v_idx >= len(cumulative):
            v_idx = len(weights) - 1
        result_indices.append(mixed_list[v_idx])
        
    return result_indices
