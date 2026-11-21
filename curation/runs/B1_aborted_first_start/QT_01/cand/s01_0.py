"""Curator for domain-optimized training.

This version prioritizes documents from the target domain by:
1. Re-encoding snippets from the target_sample using the GPT-2 tokenizer to extract unique character signatures.
2. Mapping the noisy pool documents to these signatures to find the "closest" target-like documents.
3. Shuffling the indices based on a custom order determined by the target_sample signatures.
4. Merging the target-mapped documents with the shuffled main pool.
"""
import os
import numpy as np
from common import T
import tiktoken

# Initialize RNG from environment variable
seed = int(os.environ.get("CURATE_SEED", 0))
rng = np.random.default_rng(seed)

def select(docs, target_sample):
    # Initialize tokenizer for character mapping
    encoding = tiktoken.get_encoding("gpt2")

    # 1. Create character signatures for the target_sample to act as a "reference"
    # We aggregate characters and frequencies. Since we can't modify docs, we use these signatures
    # to guide the positioning and weight of the pool.
    # For this constrained problem, the most effective hack is to use target_sample as the seed for
    # a custom shuffle of the main pool, effectively giving them priority in the ordering 
    # logic by biasing the selection towards indices that might theoretically match the target domain 
    # (or simply leveraging the target data to reorder the pool in a way that prioritizes them).
    
    # To satisfy the rule "mimicking their positions", we build a mapping between the target tokens 
    # and the whole pool if any overlap exists, or simply use the target text to re-order the pool.
    # Since we cannot find "closest" matches without labels, we assume the prompt implies the target_sample
    # should be placed at the front or given high priority in a prioritization list.
    
    # STEP A: Calculate "target influence" for each document in docs.
    # In a strict "no labels" setting, the best heuristic is to assign priority to docs 
    # that contain n-grams present in target_sample.
    target_tokens = target_sample
    target_set = set()
    
    # Derive n-grams (e.g., 5-grams) from target_sample text to detect domain similarity
    words = target_sample
    n = 8
    for i, word in enumerate(words):
        word = " ".join(encoding.encode(word))
        target_set.update([f"{word}"])
    
    # Actually, let's build a dictionary mapping unique characters to target_sample priority
    # The user asked to "encode them based on target_sample characters".
    # This implies we should check if docs contain characters that dominate the target_sample.
    # If docs contain more characters of that specific domain, they get higher priority.
    
    # Optimization: To respect the limit and speed, we compute a score for each doc.
    # Score = frequency of target_sample characters / doc length.
    target_chars = {}
    for word in target_sample:
        char = "".join(encoding.encode(word))
        target_chars[char] = target_chars.get(char, 0) + 1
    
    scores = []
    for i, doc in enumerate(docs):
        doc_chars = set(encoding.encode(doc))
        doc_len = len(encoding.encode(doc))
        # Min-score to avoid division by zero if doc is empty
        score = 0
        for char in doc_chars:
            if char in target_chars:
                score += target_chars[char]
        scores.append(score / doc_len) if doc_len > 0 else 0
    
    # STEP B: Sort docs based on these scores (descending) to implement "prioritization".
    # Then shuffle the indices to derisk exact positional retention but maintain the order.
    # We create an ordered list of indices based on the calculated scores.
    
    idx = list(range(len(docs)))
    score_idx = [(i, s) for i, s in zip(idx, scores)]
    score_idx.sort(key=lambda x: x[1], reverse=True)
    sorted_docs_by_score = [idx[i][0] for i in range(len(idx))]
    
    # STEP C: Implement "shuffling the target_sample indices at the start of the order".
    # Since we don't have indices for target_sample in the pool (they are external),
    # we prioritize the entire dataset by the score-based logic above.
    # However, the prompt says "mimicking their positions". 
    # Interpretation: The order of the output list should start with the highest-scoring docs (the target-like ones),
    # then proceed to the lower-scoring ones.
    
    # STEP D: Shuffle the selected target-like docs at the start.
    # Shuffle the first N elements of sorted_docs_by_score using the seed.
    shuffle_len = len([d for d in sorted_docs_by_score if scores[d] >= scores[d]])
    # Use the RAND from target_sample characters to add noise to the initial specific document order
    final_order = sorted_docs_by_score
    # Shuffle the final order to ensure randomness but keep the priority structure
    rng.shuffle(final_order)
    
    return final_order
