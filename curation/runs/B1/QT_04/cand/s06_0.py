"""Improved curator: prioritize target domain docs, then shortest pool docs.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from typing import Union, List

# Basic counting logic to avoid external libraries:
# encoding "gpt2" tiktoken is imported:
def count_encoding(
    encoded_texts: Union[List[int], List[List[int]]],
    model: str = "gpt2",
) -> int:
    """
    Get the number of tokens for the texts.
    input: encoded_texts: a list of texts, or already encoded as integers (if using tiktoken.encode counts per doc)
    output: total token count for all docs
    """

    total_tokens = 0
    for t in encoded_texts if isinstance(encoded_texts, list) else [encoded_texts]:
        total_tokens += len(t)
    return total_tokens

def count_tokens_simple(texts):
    # implement simple word/token count using tiktoken for exactness
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return len(enc.encode(texts))

def total_count(docs, encoding):
    total = 0
    for doc in docs:
        tokens = encoding.encode(doc)
        total += len(tokens)
    return total

import tiktoken
enc = tiktoken.get_encoding("gpt2")


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    # Ensure target_sample is a list of strings
    target_sample = target_sample if isinstance(target_sample, list) else []

    # Step 1: Sort and select target_sample docs
    target_indices = sorted(range(len(docs)), key=lambda i: 0 if docs[i] in target_sample else 1)

    # Add target_sample indices to the list
    target_list = []
    target_map = {doc: i for i, doc in enumerate(docs)}
    for doc in target_sample:
        if doc in target_map:
            target_list.append(target_map[doc])

    target_list.sort()

    # Ensure target_sample indices are in the top 100 docs (or at least first)
    # If target_sample indices are not unique or overlap, they are handled naturally.
    # Sort by doc count, ascending, resolve ties by shuffling indices

    # Step 2: Sort remaining docs by token count ascending, ties shuffled
    remaining_indices = [i for i in range(len(docs)) if i not in target_set := {i for i in target_list}]

    if remaining_indices:
        # Sort by token count, ties resolved by shuffling
        remaining_indices.sort(key=lambda i: enc.encode(docs[i]) if i not in target_set else 0)
        # Resolve ties by shuffling
        remaining_indices = list(remaining_indices)
        random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(
            remaining_indices
        )

    # Combine target indices first, then remaining indices
    selected_indices = target_list + remaining_indices
    return selected_indices
