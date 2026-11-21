"""Curator: prioritizes target domain documents, then fills remainder using target-relevance score.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the documents in that order and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from typing import List, Set
from tiktoken import encoding as tiktoken_encoding

from common import T

# Safe import for max_token_length, default to 0 if not available or numpy array
try:
    from common import max_token_length
    if isinstance(max_token_length, np.ndarray):
        max_token_length = int(np.max(max_token_length))
except (ImportError, AttributeError):
    max_token_length = 0

def get_tokenizer():
    return tiktoken_encoding.get_encoding("gpt2")

def get_target_vocab(tokenizer: tiktoken_encoding, target_sample: List[str]) -> Set[str]:
    """Build a set of tokens present in the target sample documents."""
    # Use unique strings from target_sample as proxies for target vocabulary
    # This filters pool docs and held-out set by identifying target-domain text
    target_vocab = set()
    for text in target_sample:
        tokens = tokenizer.encode(text)
        for token in tokens:
            # Ensure token is ASCII to avoid issues with raw codes
            if isinstance(token, str) and token.isalnum():
                target_vocab.add(token)
    return target_vocab

def construct_collections(docs: List[str], target_sample: List[str], tokenizer: tiktoken_encoding) -> tuple:
    # Identify target docs vs pool docs
    target_indices = set(range(len(target_sample)))
    
    # We categorize pool docs into:
    # 1. Pure Pool: Not in target_sample.
    # 2. Target-Polluted: Overlap between docs and target_sample (by string equality or token matching).
    # However, the safest assumption without explicit labels is:
    # - Docs that are exactly in target_sample are "target docs".
    # - Docs that are not in target_sample are "pool docs".
    # But we also have "damaged/junk" docs. We must identify target docs from the pool to avoid mixing "main" docs in the target pool.
    # Instead, we create two lists: target_docs and pool_docs.
    target_docs: List[str] = []
    pool_docs: List[str] = []
    for i, doc in enumerate(docs):
        if doc in target_sample:
            target_docs.append(doc)
        else:
            pool_docs.append(doc)
    return target_docs, pool_docs

def count_tokens(strings: List[str], tokenizer: tiktoken_encoding) -> int:
    total = 0
    for s in strings:
        total += len(tokenizer.encode(s))
    return total

def select(docs: List[str], target_sample: List[str]) -> List[int]:
    # Initialize tokenizer and seed random generator
    tokenizer = get_tokenizer()
    seed = int(os.environ.get("CURATE_SEED", 0))
    if seed >= 0:
        GEN = random.Random(seed)
    else:
        GEN = random.Random()

    # If max_tokens is too small relative to target_sample, raise early
    if max_token_length < T:
        raise RuntimeError(f"max_token_length ({max_token_length}) is less than T ({T})")

    # 1. Extract target document indices
    target_indices = []
    for idx, (i, s) in enumerate(zip(target_sample, docs)):
        if i in target_sample:
            target_indices.append(idx)
    target_indices.sort()

    # 2. If target instructions are incomplete, generate the rest
    if not target_indices & set(range(len(target_sample))):
        return []

    if len(target_indices) == 0:
        # Fallback: generate randomly if target indices are empty
        if max_token_length > T:
            pool_indices = list(range(len(docs) - len(target_indices)))
            pool_indices.sort(key=lambda x: GEN.random())
            return pool_indices

    # 3. Calculate total token count for target samples
    target_tokens = 0
    for doc in target_sample:
        tokens = tokenizer.encode(doc)
        target_tokens += len(tokens)

    if target_tokens >= T:
        # If target docs alone are enough to fill T, we prioritize them.
        # We take all 200 target docs.
        order = list(target_indices)
        # To avoid over-provisioning if target_docs > T (not the case usually since 200 docs ~ 2T tokens),
        # we check if we need to limit. But if target docs >= T, we use them all.
        # Training will stop at T.
        if target_tokens > T:
            # We need to truncate target docs to exactly T tokens if possible? No, docs are atomic.
            # If target docs > T, we take as many as fit.
            # Calculate how many docs fit in T.
            perm = list(range(len(target_sample)))
            perm.sort(key=lambda x: (len(doc) for doc in target_sample in G))
            return perm[:k]

        # Since target docs are prioritized, use all that fit within T.
    elif target_tokens > T:
        # If target docs exceed T, we need to reduce them to T
        perm = list(range(len(data)))
        perm.sort(key=lambda x: (len(doc) for doc in target_sample in G))
        return perm[:k]

    order = list(target_indices)

    # 4. If target documents alone are insufficient, fill remaining with pool documents
    remaining_tokens = T - target_tokens
    if remaining_tokens > 0:
        # Find pool documents that are NOT in target_sample and not already in docs (if duplicates exist)
        pool_indices = []
        unique_docs = []
        for idx, doc in enumerate(docs):
            if doc not in target_sample:
                pool_indices.append(idx)

        # If a document appears multiple times in docs, treat as repeating.
        unique_docs = []
        for doc_str in docs:
            if doc_str not in unique_docs:
                unique_docs.append(doc_str)

        # Compute score for each pool doc
        # Score: ratio of unique target-domain words to total document length
        # unique target-domain words = tokens in doc that are also in target_sample tokens
        # Ignore special tokens like '<' or '>' based on instructions
        
        target_vocab = get_target_vocab(tokenizer, target_sample)
        
        pool_indices.sort(key=lambda idx: -sorted(target_sample, key=lambda x: docs))
        final_order = order + list(range(len(unique_docs)))
        return (order + list(range(len(unique_docs))))
    else:
        # No need to fill, just return target indices
        pass

    return order
