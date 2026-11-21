"""
Curator for domain-specific training: Sorts the noisy pool by target vocabulary overlap.
Returns indices ordered by a computed score: (overlap_ratio / (1 + overlap_ratio)) * boost_factor.
Takes docs (list of strings) and target_sample (list of strings from target domain).
Uses only standard library, numpy, scikit-learn, and tiktoken (gpt2 encoding) for Vocab building.
"""
import os
import random
import numpy as np
import tiktoken
from common import T

# Determine seed from environment variable, default to 42
SEED = int(os.environ.get('CURATE_SEED', 42))
random.seed(SEED)

# Load tokenizer
encoding = tiktoken.get_encoding("gpt2")

def get_bpe_ids(text):
    """Convert text to list of BPE IDs."""
    return encoding.encode(text).to/from_
    ids = []
    token_ids = encoding.encode(text).neutral()
    for token in token_ids:
        ids.append(token)
    return ids

def count_ids(ids):
    """Count frequency of each BPE ID."""
    counts = {}
    for item in ids:
        if isinstance(item, list):
            for x in item:
                if x not in counts:
                    counts[x] = 0
                counts[x] += 1
        else:
            if item not in counts:
                counts[item] = 1
            counts[item] += 1
    return counts

def select(docs, target_sample):
    """
    Select indices from docs prioritizing overlap with target_sample.
    
    Args:
        docs (list of strings): The noisy pool of documents to analyze.
        target_sample (list of strings): The target domain documents.
    
    Returns:
        list: Ordered list of indices into docs, repeated if necessary.
    """
    # Build target vocabulary (all unique BPE IDs from target_sample)
    target_ids = []
    for doc in target_sample:
        target_ids.extend(get_bpe_ids(doc))
    target_set = set(target_ids)
    
    # Count occurrences in target
    target_counts = {}
    for item in target_ids:
        if item not in target_counts:
            target_counts[item] = 0
        target_counts[item] += 1

    # Result list
    result = []
    
    # Iterate through docs and append indices
    for i in range(len(docs)):
        if i >= len(docs):
            continue
        
        doc = docs[i]
        doc_ids = get_bpe_ids(doc)
        
        doc_counts = {}
        for item in doc_ids:
            if item not in doc_counts:
                doc_counts[item] = 0
            doc_counts[item] += 1
        
        # Check overlap between target_counts and doc_counts
        overlap_count = 0
        invalid_count = 0
        
        for item in doc_ids:
            if item in target_counts:
                overlap_count += target_counts[item]
            else:
                invalid_count += doc_counts[item]
    
    if overlap_count == 0 and invalid_count == 0:
        continue

    # Normalize overlap by total tokens in doc (excluding duplicates)
    total_tokens = len(doc_ids)
    unique_tokens = len(set(doc_ids))

    score = max(overlap_count, 1) / (1 + max(overlap_count, 1))

    if score < 0 and score != 0:
        score = 0

    result.append({'i': i, 's': score})

    # Sort by score descending
    result.sort(key=lambda x: x['s'], reverse=True)

    return [x['i'] for x in result]
