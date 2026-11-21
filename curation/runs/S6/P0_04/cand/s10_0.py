"""Curated curator: Trigram-based structural overlap.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding_for_model("gpt2")
    
    # Collect tokens from the target sample
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc))
    
    # Identify common contiguous trigrams in the target sample
    # We treat the target sample as a continuous stream of tokens to find patterns
    trigrams = set()
    for i in range(len(target_tokens) - 2):
        trigrams.add((target_tokens[i], target_tokens[i+1], target_tokens[i+2]))
        
    # Map each starting token of a target trigram to a score based on how many
    # trigrams it starts. This emphasizes tokens that participate in target structures.
    trigram_counts = Counter(trigrams)
    # A token is "valuable" if it is the start of a frequent trigram
    valuable_starts = {}
    for trigram in trigrams:
        first_token = trigram[0]
        valuable_starts[first_token] = valuable_starts.get(first_token, 0) + 1
        
    # Score documents based on the proportion of tokens that are starts of target trigrams
    # This rewards documents that share the local sequential structure of the target domain.
    doc_scores = []
    for doc in docs:
        tokens = enc.encode(doc)
        if not tokens:
            doc_scores.append(0.0)
            continue
            
        match_count = 0
        for t in tokens:
            if t in valuable_starts:
                match_count += valuable_starts[t]
        
        # Score is the density of valuable structure relative to doc length
        score = match_count / len(tokens)
        doc_scores.append(score)
        
    # Rank indices: primary key is score, secondary is original index (to keep it stable)
    indices = list(range(len(docs)))
    # We want highest scores first. Using negative index as a tie-breaker for deterministic behavior.
    indices.sort(key=lambda i: (doc_scores[i], -i), reverse=True)
    
    return indices
