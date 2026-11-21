"""
Curator that prioritizes target domain documents then cleans the pool by
frequency of classes in the target domain (SCLC-like logic).

Usage:
    Training will first load the 200 target documents, then the remaining
    documents are passed through a frequency-based filter that retains
    only those with high tokens-per-class frequency.

This file is the only one to change.
"""
import os
import random
from collections import Counter

from tiktoken import Encoding
from common import T

# Maximum number of tokens that can be used from the "useful" pool (target sample)
K = 60000  # sarsa version
# Number of tokens allowed from the pool per source
M = 12000  # max tokens per source

# SCLC mutation rate
U = 0.01  # < K

# How to "clean" the corpus
# SCLC: semi-length free clean corpus

# Fixed encoding parameters for GPT-2
ENC = Encoding(pat_str="default", mergeable_ranks=12705, special_tokens=["<|endoftext|>"])


def _count_tokens(s):
    """Count token size of a string (including junk)"""
    # tiktoken.encoding.encode returns a list of tokens (integers)
    return len(ENC.encode(s.strip()))


def select(docs, target_sample):
    """
    Curates the document pool by:
      1. Prioritizing the 200 target domain documents first.
      2. Sorting the remaining documents by their 'class' frequency
         to ensure only high-quality, clean text is used, without damaging
         target domain vocabulary.
    The seed for randomness is set from the CURATE_SEED environment variable.
    """
    # Deterministic seed
    rd = random.Random(int(os.environ.get("CURATE_SEED", 0)))

    result = []

    # 1. Identify and prioritize target domain documents
    target_matches = []
    for i, doc in enumerate(docs):
        if doc in target_sample:
            target_matches.append(i)

    result.extend(target_matches)

    # Remove target indices from consideration to find remaining docs
    target_indices_set = set(target_matches)
    remaining_docs = [i for i in range(len(docs)) if i not in target_indices_set]
    remaining_docs_list = [docs[i] for i in remaining_docs]

    # 2. Sort remaining documents by frequency of their tokens (SCLC logic)
    # We group tokens by their frequency across the document to estimate "class" cleanliness.
    # Documents with higher unique token frequencies (and thus lower repetition/junky tokens) are prioritized.
    doc_freq_scores = []
    
    # Compute frequency scores for each remaining document
    for doc in remaining_docs_list:
        tokens = ENC.encode(doc.strip())
        token_counts = Counter(tokens)
        # Score: number of unique tokens (approximates vocabulary richness/cleanliness)
        # Higher score = more varied/likely cleaner text, less likely junk/duplicates
        unique_tokens = len([t for t, c in token_counts.items() if c == 1])
        doc_freq_scores.append((len(remaining_docs_list), doc, unique_tokens))
    
    # Sort by unique token count (descending), then by length (descending) to prioritize diverse, long docs
    sorted_remaining = sorted(doc_freq_scores, key=lambda x: (x[2], x[1]), reverse=True)

    # Extract indices corresponding to the sorted high-freq docs
    final_remaining_indices = [index for index, _, _ in sorted_remaining]

    # Combine target and remaining
    result.extend(final_remaining_indices)

    # Shuffle the result based on the seed
    rd.shuffle(result)

    return result
