import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Pre-encode target sample to get a global set of tokens
    # Using a set of tokens is faster than calculating Jaccard for each doc
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))

    # Score documents based on how many tokens they share with the target sample
    # We use a simple overlap count to save computation time.
    scores = []
    for doc in docs:
        tokens = enc.encode(doc)
        doc_set = set(tokens)
        # Intersection size is a proxy for relevance
        score = len(doc_set.intersection(target_tokens))
        scores.append((score, len(tokens)))

    # Create list of (index, score)
    # We want to prioritize documents with high overlap.
    # To ensure we don't get stuck in a loop of tiny junk docs with high overlap,
    # we can weight by length or just take the best.
    indexed_scores = []
    for i in range(len(docs)):
        # Score = overlap / total_tokens_in_doc to normalize, 
        # but raw overlap is often better for retrieval.
        # Let's use a small penalty for very short docs to avoid junk.
        s, l = scores[i]
        if l < 10:
            indexed_scores.append((i, s - 10))
        else:
            indexed_scores.append((i, s))

    # Sort by score descending
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract indices
    best_indices = [idx for idx, score in indexed_scores]
    
    # We need to return a list that fills T tokens or is large enough 
    # for the trainer to cycle efficiently.
    # Since T is 9,830,400, and docs are ~2T in pool, we want a good 
    # diversity of the best ones.
    # The trainer cycles the list. If we provide 500 diverse good docs, 
    # the trainer will see them multiple times.
    return best_indices[:500]
