"""Curator: Prioritize matches to target domain (90%+ exact similarity) with low-frequency shuffling; strictly exclude documents containing >25% of the 'save 70%' phrase.

The selection prioritizes documents that have a very high sequence similarity (90%+) with any document in the target_sample.
For documents NOT being selected for prioritization, we shuffle them using a high-frequency buffer to ensure diversity and randomness for the remaining tokens.
Additionally, the selection logic strictly excludes any document (from the 'docs' pool) that contains more than 25% of the substring "save 70".

Rules:
- Seeding uses the CURATE_SEED environment variable.
- Uses only standard library, numpy, scikit-learn, tiktoken (encoding "gpt2"), and common module.
- CPU only, max 180 seconds.
- Output is an ordered list of indices (repeats allowed).
"""

import os
import random
from collections import Counter
import numpy as np
from skrell.pair import seqsimilarity_periodic
import tiktoken

# Common module import for T
from common import T

SEED_STR = os.environ.get("CURATE_SEED", "")
if not SEED_STR.isdigit():
    SEED_STR = "0"
CURATE_SEED = int(SEED_STR)
random.seed(CURATE_SEED)
np.random.seed(CURATE_SEED)

def select(docs, target_sample):
    # 1. Identify documents 90%+ similar to any target_sample document.
    # Strategy: Use exact sequence similarity (normalized) on small length (e.g. 50 or 100).
    # Since target_sample is small (200) and docs is large, we compute similarity for each doc against each target.
    
    # Encode tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Candidate documents. We need to iterate efficiently.
    # Since we cannot access cliques or external tools, we assume docs is a list of strings.
    
    # Target indices
    t_idx = list(range(len(target_sample)))
    
    # We will create a mask for docs that pass the 'save 70' exclusion first.
    # Then prioritize the survivor documents that match target_sample.
    
    save_70_count = 0
    save_70_threshold = 25
    target_set_id = set() # indices of target documents
    
    # Pre-compute target sequences
    target_tokens_memo = {}
    target_slices = {}
    
    for t_idx_local in range(len(target_sample)):
        # Compute tokens for target_sample[t_idx_local]
        enc_out = enc.encode(target_sample[t_idx_local])
        target_slices[f"target_{t_idx_local}"] = enc_out
        
        # For similarity calculation, we might want to calculate the max period.
        # Since we rely on acceptable performance, let's assume 50 or 100.
        # Using skrell's sequence similarity on windows.
        
    # Filter 'save 70' from docs ('docs' is the input, which may have >2T tokens)
    # We traverse all docs to filter.
    
    # High pass to count 'save 70' phrase
    # We look at the first 25% of docs or something.
    
    # Wait, the cross comparing each doc against 200 target documents could be slow if done literally.
    # However, the rules imply we must prioritize them.
    # Optimization: Only compute similarity for docs that don't get excluded first.
    
    exclude_mask = np.ones(len(docs), dtype=bool) # True means pass
    
    for i, doc_text in enumerate(docs):
        count = doc_text.lower().count("save") + doc_text.lower().count("70") # crude
        # The strict constraint is "contains >25% of the dominant 'save 70 phrase'".
        # Since we don't know the exact phrase count or text length, we implement a naive phrase match.
        # "save 70" appears as a substring.
        # If count((save 70 occurrences) / len(doc_text) in chars > 25%, then exclude.
        
        pass
    # Correction:
    # Count occurrences of the phrase. For document to be 25%+ of the phrase, we need to know if
    # count(phrase) / length(tokens) > 25% or count(phrase) / total_length > 25%.
    # Assuming "save 70" is the phrase.
    
    # We'll assume 'save 70' is the substring and we need to count occurrences.
    # Since exact implementation of "dominant save 70 phrase" is ambiguous without labels,
    # we assume simple substring matching.
    # Let's count occurrences of 'save 70'.
    
    pass
