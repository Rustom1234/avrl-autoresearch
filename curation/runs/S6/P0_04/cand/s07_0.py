"""Curated curator: Lexical centering based on target token frequency.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import collections
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to tokenize both the pool and the target sample
    # To minimize overhead, we only need the tokens of the target sample
    enc = tiktoken.encoding.get_encoding("gpt2")
    
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc))
    
    # Count absolute frequency of target tokens across the pool
    # We need to identify which docs contain which target tokens
    token_to_doc_indices = collections.defaultdict(list)
    
    # To handle the 2T size efficiently, we iterate once
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        # Use a set for tokens in each doc to avoid over-counting 
        # occurrence of the same token within one doc
        unique_tokens = set(tokens)
        for t in unique_tokens:
            token_to_doc_indices[t].append(i)
            
    # Calculate absolute frequency of every target token in the pool
    # (Implicitly handled by token_to_doc_indices)
    # Sort target tokens by their frequency in the pool
    target_counts = Counter()
    for t in target_tokens:
        # Every time a target token appears in a doc, it contributes to count
        # However, we want the frequency of tokens in the pool.
        # Since we only have the pool, we count how many docs contain it 
        # as a proxy for frequency if we can't iterate twice.
        pass

    # Actually, to follow "absolute frequency of every target_sample token in the pool":
    # We must count every occurrence in the pool.
    pool_token_counts = Counter()
    for doc in docs:
        pool_token_counts.update(enc.encode(doc))
    
    # Now we rank target tokens by their absolute frequency in the pool
    # Get unique target tokens
    unique_target_tokens = list(set(target_tokens))
    # Sort target tokens by frequency descending
    unique_target_tokens.sort(key=lambda t: pool_token_counts[t], reverse=True)
    
    # Score docs by "relative overlap with the most frequent target tokens"
    # We sum the ranks of the tokens present in the doc that are in our target list
    # A higher score means the doc contains more of the "important" target tokens.
    doc_scores = [0.0] * len(docs)
    
    # Map token to its weight (higher frequency = higher weight)
    # Let's use the frequency itself as the weight
    for rank, t in enumerate(unique_target_tokens):
        freq = pool_token_counts[t]
        # We want to prioritize docs containing tokens with high freq.
        # Let's use a weight: high frequency tokens get higher weight.
        # The prompt says "sort documents by their relative overlap with 
        # the most frequent target tokens".
        # We can use a weight that decays with rank or just use frequency.
        # Using frequency directly:
        # Each doc gets points for every target token it contains.
        # The weight of that point is the frequency of that token in the pool.
        pass
    
    # Re-evaluating: The most efficient way to "sort by overlap with most frequent"
    # is to score each doc: Score = sum(frequency(t) for t in (tokens_in_doc AND target_tokens))
    # but we want to weight the "most frequent" higher.
    # Let's use a weight: weight = 1 / (rank + 1) or similar? 
    # "Relative overlap with the most frequent" usually implies prioritizing the top ones.
    
    # Let's use the frequency itself as the weight for the overlap.
    weights = {}
    for i, t in enumerate(unique_target_tokens):
        # Weighting by frequency, but also ensuring the "most frequent" (lowest rank) 
        # has a significant impact.
        weights[t] = pool_token_counts[t]

    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        score = 0
        for t in doc_tokens:
            if t in weights:
                score += weights[t]
        doc_scores[i] = score
        
    # Sort indices by score descending
    indices = list(range(len(docs)))
    # Sort by score (primary) and then original index (secondary) to stay stable
    indices.sort(key=lambda i: (doc_scores[i], -i), reverse=True)
    
    return indices
