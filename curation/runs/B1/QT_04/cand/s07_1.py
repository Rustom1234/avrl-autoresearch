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

    # For now, we are using a dictionary-based approach; this will be done with numpy vectors if possible.
    # We will cross-reference the n-gram presence (simplified to token presence) in the target vocabulary.

    doc_target_counts = defaultdict(lambda: 0)

    for doc_idx, doc_str in enumerate(docs):
        # Skip target samples in this loop (if targets overlap, handle separately)
        if doc_idx >= len(target_sample):
            continue
        # Note: We iterate over all docs including target samples to build global token counts if necessary, 
        # but the logic below focuses on non-target docs. Let's restructure to just process non-targets.
        pass

    # Re-approach: Only process non-target docs for ranking
    remaining_indices_all = [(i, docs[i]) for i in range(len(docs)) if i >= len(target_sample)]

    # Dictionary to store raw token scores based on target vocab overlap
    raw_scores = {}

    target_doc_indices = set(range(len(target_sample)))

    for i, doc_str in remaining_indices_all[0]:  # Use enumerate on the filtered list logic manually for clarity or just loop
        if i in target_doc_indices:
            continue
            
        # Process only non-target documents
        pass

    # Correct loop for non-target documents
    for doc_idx in range(len(docs)):
        if doc_idx < len(target_sample):
            continue
        
        if len(docs[doc_idx]) == 0:
            # If document is empty, score 0
            raw_scores[doc_idx] = (0, 0)
            continue

        tokens = enc.encode(docs[doc_idx])
        
        if not tokens:  # No tokens to count
            raw_scores[doc_idx] = (0, 0)
            continue

        target_tokens_count = sum(1 for token_id in tokens if token_id in target_vocab)
        total_tokens_count = len(tokens)
        
        raw_scores[doc_idx] = (target_tokens_count, total_tokens_count)

    # We will now construct a weighted score for the remaining documents

    # Calculate final ratio scores
    scored_docs = []
    for doc_idx in range(len(docs)):
        if doc_idx < len(target_sample):
            continue
        
        target_count, total_count = raw_scores[doc_idx]
        if total_count > 0:
            ratio = target_count / total_count
        else:
            ratio = 0.0
        
        scored_docs.append((doc_idx, ratio, total_count))

    # Sort by ratio descending, then by total tokens descending (to prioritize more significant documents)
    scored_docs.sort(key=lambda x: (-x[1], -x[2]))
    sorted_remaining_ids = [x[0] for x in scored_docs]

    #
    # Step 3: Assemble the final ordered list
    if not sorted_remaining_ids:
        # No remaining documents, just return target indices (if any)
        return target_indices

    # Final list: target documents first, then sorted remaining documents
    final_list = target_indices + sorted_remaining_ids

    return final_list
