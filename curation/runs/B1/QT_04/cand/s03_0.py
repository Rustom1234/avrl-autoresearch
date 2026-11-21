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

ENC = Encoding("gpt2")


def _count_tokens(s):
    """Count token size of a string (including junk)"""
    return len(ENC.encode((((s) or "").strip(),)))


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

    # Prioritize target domain documents
    target_idx = set(range(len(target_sample)))
    target_indices = [i for i, doc in enumerate(docs) if any(doc == t.strip() for t in target_sample)]
    # Check if target documents are present
    if not target_sample:
        target_sample = [doc for doc in docs[:200] if all(doc != target) for target in target_sample]
        target_indices = list(range(len(target_sample)))

    # Add any repeated documents from the target sample
    result = []
    for i, doc in enumerate(docs):
        if doc in target_sample:
            result.append(i)

    # Remove duplicates and unused from the pool
    used_indices = set(target_indices)
    remaining_docs = [doc for i, doc in enumerate(docs) if i not in target_indices]

    # Sort the remaining documents by frequency of their classes (SCLC)
    # Compute frequency by token class (approximate via word frequency)
    class_freq = Counter()
    for doc in remaining_docs:
        tokens = ENC.encode(doc.strip())
        # Group tokens by class (approximate by using frequency of unique tokens)
        freq = Counter(tokens)
        class_freq += freq

    # Sort by frequency of classes (descending)
    sorted_docs = sorted(remaining_docs, key=lambda d: class_freq[d], reverse=True)
    # Then, filter out documents with low class frequency
    # This ensures only high-quality, clean text is used
    r = []
    for doc in sorted_docs:
        if class_freq[doc] > 0:
            r.append(doc)

    # Combine target and remaining
    result.extend(r)
    result = [docs[i] for i in result]

    # Shuffle the result
    rd.shuffle(result)

    return result
