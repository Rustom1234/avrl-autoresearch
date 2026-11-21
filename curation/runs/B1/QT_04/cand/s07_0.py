"""Prioritized content curator: start with target samples, then rank remaining docs by target vocabulary overlap.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from collections import defaultdict
from common import T

# Prerequisites: tiktoken for tokenizer, numpy for vectorized operations or dictionary fallback.
# We will use tiktoken for encoding and a simple tokenization strategy.

# We will need to use a list of 200 target strings. They are not in the pool and not in held-out set.
# To implement the cross-entropy (turbo.ReformatDocs) logic in a simplified way:
# 1. We get the target vocabulary
# 2. For each token in the dataset, we calculate the presence in the target vocabulary.

# Encodings from the standard tokenization module
enc = None


def select(docs, target_sample):
    global enc

    # Set seed for reproducibility from the environment variable
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)

    # Step 1: Include the target documents first.
    target_indices = list(range(len(target_sample)))

    #
    # Step 2: Rank remaining docs by their token import from target vocabulary.
    # We will use tiktoken to count the presence of each token's tokenization mapping to the target vocabulary.

    if enc is None:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

    #
    # Calculate tokenization intersection for target_sample
    #target_vocab = set()
    # We can use a set or a dictionary to store the tokens per target document.
    target_vocab = set()
    for t_str in target_sample:
        tokens = enc.encode(t_str)
        target_vocab.update(tokens)

    # We will create a token distribution dictionary that maps token ID to count

    remaining_indices = [i for i in range(len(docs)) if i >= len(target_sample)]
    # For now, we are using a dictionary-based approach; this will be done with numpy vectors if possible.
    # We will cross-reference the n-gram presence (simplified to token presence) in the target vocabulary.

    token_counts = dict()

    # We will create a global accumulation of token counts for each document
    # doc_token_counts[i] = { token_id: count }
    # Then, we will count overlap with target_vocab

    doc_target_counts = defaultdict(lambda: defaultdict(int))

    for doc_idx, doc_str in enumerate(docs):
        # Skip target samples in this loop (if targets overlap, handle separately)
        if doc_idx < len(target_sample):
            continue
        tokens = enc.encode(doc_str)
        for token_id in tokens:
            if token_id not in token_counts:
                token_counts[token_id] = 0
            token_counts[token_id] += 1

    # We will now construct a weighted score for the remaining documents

    # For each remaining document, we count how many tokens fall into the target_vocab set
    # and calculate the ratio: target_tokens / total_tokens
    sorted_remaining = []
    for doc_idx in range(len(docs)):
        if doc_idx < len(target_sample):
            continue
        # If the document has no tokens (should not happen), skip
        if len(doc_str) == 0:
            sorted_remaining.append((doc_idx, 0.0, 0.0))
            continue

        tokens = enc.encode(docs[doc_idx])
        if no tokens:
            continue

        target_tokens_count = 0
        for token_id in tokens:
            if token_id in target_vocab:
                target_tokens_count += 1

        total_tokens_count = len(tokens)
        if total_tokens_count > 0:
            ratio = target_tokens_count / total_tokens_count
            # Score: higher is better for target overlap
        else:
            ratio = 0.0

        sorted_remaining.append((doc_idx, ratio, target_tokens_count))

    # Sort by ratio descending, then by total tokens descending (to prioritize more significant documents)
    sorted_remaining.sort(key=lambda x: (-x[1], -x[2]))
    sorted_remaining = [x[0] for x in sorted_remaining]

    #
    # Step 3: Assemble the final ordered list
    if not sorted_remaining:
        # No remaining documents, just return target indices (if any)
        return target_indices

    # Final list: target documents first, then sorted remaining documents
    final_list = target_indices + sorted_remaining

    return final_list
