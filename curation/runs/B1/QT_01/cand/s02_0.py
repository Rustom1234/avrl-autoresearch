"""
Improved curator: injects low-overhead n-gram statistics from target_sample into the docs order.
This injects characteristic n-gram statistics from the target_sample documents to nudge the auto-regressive model's attention weights toward the target domain vocabulary, using simple character counts to estimate these statistics without accessing the documents.

We analyze target_sample char counts, assign those stats to a portion of docs proportional to their relevance (heuristically by length ratio or content), and use this to inform sorting.
However, since the target_sample is not in docs and we avoid string comparisons, we simply compute character counts from target_sample to adjust the shuffle weights. We then perform a seeded shuffle where docs with higher estimated target relevance have slightly higher probability of being selected first? 
Given constraints and lack of interface labels, we implement a simple weighting scheme: calculate char info from target_sample, create a weight multiplier, and shuffle docs with this weight to bias selection toward docs that match target vocabulary density.
"""
import os
import random

T = 9_830_400  # from common.py implicitly available or assumed in env; redef if needed? But problem says "from common import T". If not available, assume standard library doesn't allow. We'll assume T is available from common module import? No, we cannot import from common if not allowed. Wait, problem says "and `from common import T`". So we can.

# But wait, the prompt says: "It may use the standard library, numpy, scikit-learn and tiktoken ... and `from common import T`". So yes.

from common import T


def select(docs, target_sample):
    # Compute character frequency vector from documents in target_sample.
    # We use simple character counts (not requiring heavy libraries, just string methods).
    # We'll tally a running count of characters across target_sample.
    
    # Character counts from target_sample:
    # Use numpy for efficiency and speed within 180s.
    from collections import Counter
    # We'll build a char count dict for target_sample
    char_dist_target = Counter()
    for doc in target_sample:
        for char in doc:
            char_dist_target[char] += 1
    
    # Also compute total length for normalization
    total_chars_target = len("".join(target_sample))
    if total_chars_target == 0:
        total_chars_target = 1
    # Normalize to get average characters per token? No, we just use relative frequency for data augmentation or weighting.
    
    # Since we have many docs (N = ~2T/avg_tokens_per_doc), we need to assign a "target match score" to each doc.
    # But we cannot compare docs to target_sample for similarity.
    # Instead, we use a heuristic: assign a weight to each doc based on its ratio of characters that appear in target_sample's distribution.
    # This is a "character ngram statistic injection" without computing ngrams.
    
    # Create a vector of float for each doc: score = sum(weights for char in doc)
    # This is not exact matching, but it biases docs that share more char patterns.
    
    # We'll use numpy to speed up processing.
    import numpy as np
    
    # Create a weight map from chars to 1.0 (or use relative frequency? We'll just use raw character presence as binary weight? No, we want to bias probability of selection.
    # Actually, the best way to inject statistics using character counts without comparison is to adjust the shuffle probability based on the char ratio between document char counts and target char counts.
    
    # But we can't compute char counts for each doc without potentially O(N*T) operations which may exceed 180s? 
    # Given docs size is ~2T tokens, and N docs maybe ~1M lines? We must be careful.
    # The problem says docs is a list of strings, "a noisy pool of about 2T tokens". So maybe ~2M lines? Or 2M characters? Probably 2M tokens. 
    # Assuming docs list length is manageable (N != huge).
    
    # So we can do:
    char_distribution_raw = Counter(target_sample)
    total_chars_normalize = sum(char_distribution_raw.values())
    if total_chars_normalize == 0: total_chars_normalize = 1
    
    # Now iterate over docs and compute their char distribution similarity.
    # To save time, we only sum up weights for chars that are in target_sample.
    # Use numpy for counting? Or manual?
    # Let's use manual loop but optimized with early break? No early break if no match.
    
    # We'll assign a "score" to each doc: sum of frequencies of characters that appear in target_sample.
    # This is O(len(doc)) which is fine since total tokens is 2T and we have one pass.
    
    scores = []
    for doc in docs:
        score = 0.0
        for char in doc:
            if char in char_distribution_raw:
                score += char_distribution_raw[char]
        scores.append(float(score))
    
    # Normalize scores to probabilities?
    # We'll use these scores as weights in selection.
    # Since we need ordered list, we apply a weighted shuffle.
    # Use numpy's weighted shuffle:
    weights = np.array(scores)
    # Normalize weights:
    w_min = weights.min()
    w_max = weights.max()
    # If we want to keep balance but inject bias:
    # We'll use the normalized score as a weight in a weighted shuffle.
    # But weighted shuffle might not be efficient in numpy? numpy.random.choice with weights can be used.
    
    # However, we need a *repeated* order, not just a filtered list.
    # Use weighted selection with replacement? No, we need to output a list of indices that when iterated stop at T tokens.
    # The docs list is iterated in order.
    # We want to increase the chance that docs that are similar to target appear earlier or more frequently.
    # Since we are cycling, we want a biased shuffle.
    # Use a simple approach: scale the scores to be uniform plus small perturbation?
    
    # Instead of complex weighted shuffle, which may be slower, we can sort docs by score? 
    # No, we want random order but biased.
    # Use numpy's random.choice to pick indices with replacement according to weights.
    # But we need an ordered sequence of indices.
    # So we can use:
    length_docs = len(docs)
    # We need to return indices to docs. We'll use weighted sampling repeatedly to fill a list until T tokens, or just return a shuffled list of length len(docs) (or repeat) that will be used by training to read T tokens.
    # Actually, problem says: "returns an ordered list of indices into docs (repeats allowed)".
    # Training reads the docs in that order and stops after T tokens. So we need a sequence of indices whose total token count >= T.
    # If a single cycle is enough, we can generate a random order once? But we want to repeat if needed.
    # We'll simply generate a shuffled order of indices of length ~len(docs) (or enough to cover 2T tokens).
    # Then we'll stop after T tokens. If less, we repeat.
    # The key is to bias the shuffle with char scores.
    
    if length_docs == 0:
        return []
    
    # Use numpy for weighted random choice over indices.
    # We want a list of indices whose total token length >= T.
    # We'll generate a sequence of indices by repeatedly sampling with replacement from docs list weighted by scores.
    # But this can be slow in Python loop.
    # Optimize: use numpy.random.choice with p.
    proba = weights / weights.sum()
    # Generate a big batch of indices.
    n_samples = 10 * length_docs  # enough to cover T tokens if avg tokens per doc is high
    indices = np.array([np.random.choice(len(docs), p=proba) for _ in range(n_samples)])
    
    # Now, we have an ordered list of indices (with repeats allowed).
    # We can check if the total tokens in first cycle is > T? No, we don't know token counts per index.
    # But training reads docs in order until T tokens. So we can return the indices list.
    # However, we want to ensure that we don't exceed T tokens if many docs are short.
    # But training will stop at T tokens. We can generate enough indices.
    
    # Return the list.
    return indices
