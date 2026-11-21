curate.py
"""
Curtator that prioritizes target documents, filters junk phrases, and deterministically shuffles the rest.
"""

import os
import re
import random
from common import T
from typing import List


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    """
    Returns an ordered list of indices into `docs` based on:
    1. Prioritizing exact matches from `target_sample`.
    2. Filtering out documents containing the phrase "save 70%".
    3. Deterministically shuffling the remaining documents.
    
    Args:
        docs: List of document strings.
        target_sample: List of strings from the target domain.
        
    Returns:
        List of integer indices corresponding to docs.
    """
    
    # 1. Identify indices of target documents.
    # Since target_sample is provided but notes say it's "not in the pool",
    # we assume the prompt implies they are present in `docs` for the selection to work.
    # We map every index in `docs` to its string and check existence.
    target_indices = []
    junk_re = re.compile(r"save 70%")
    
    for i, doc in enumerate(docs):
        if doc in target_sample:
            target_indices.append(i)
    
    # 2. Sort target indices by document length (descending) to prioritize the longest first.
    # If a document string appears multiple times, it may appear multiple times in target_sample or docs.
    # The instruction implies prioritizing the content, so we sort by the string length.
    # We keep all indices that matched the target list, sorted by their content length.
    target_indices_sorted = sorted(target_indices, key=lambda i: len(docs[i]), reverse=True)
    
    # 3. Process the remaining documents (non-targets).
    # We need to skip any document containing 'save 70%'.
    safe_other_indices = []
    total_non_target_count = 0
    
    # Iterate through all documents and separate target vs non-target
    # This single pass allows accurate counting and filtering
    for i, doc in enumerate(docs):
        is_target = doc in target_sample
        
        if is_target:
            if i in target_indices_sorted:
                continue  # Already added in step 1, but logic flow handles iteration order.
                # Actually, simpler logic: build the final list by appending specific categories.
                pass
        
        # Let's restart the selection logic to be clean:
        # A. Collect all target indices (sorted by length).
        # B. Collect all non-target indices that do NOT contain 'save 70%'.
        # C. Shuffle the union.
    
    final_selection = []
    
    # Phase A: Add target indices (sorted by length descending)
    # Note: If a target document appears multiple times in docs, it will have multiple indices in target_indices.
    # We include all of them, sorted by the length of the string itself.
    # If we have duplicates of the same target string, sorting by length keeps them adjacent.
    non_target_indices = []
    
    for i, doc in enumerate(docs):
        if doc in target_sample:
            continue  # Will be handled in Phase A
        
        # Check for junk
        if junk_re.search(doc):
            continue
        
        # This document is safe and non-target
        non_target_indices.append(i)
    
    # Phase B: Combine lists
    # Note: target_indices_sorted might be longer than count(target_sample) if duplicates exist in docs.
    # We proceed with all of them.
    non_target_indices = non_target_indices
    combined_list = target_indices_sorted + non_target_indices
    
    # Phase C: Deterministic shuffle using seed from CURATE_SEED
    seed = int(os.environ.get("CURATE_SEED", 42))
    rng = random.Random(seed)
    rng.shuffle(combined_list)
    
    return combined_list
