import os
import random
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

def select(docs, target_sample):
    """
    Ranks documents in docs by their token overlap/similarity with the target_sample
    and returns indices in descending order of similarity.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Helper to get set of unique tokens
    def get_token_set(text):
        tokens = enc.encode(text)
        return set(tokens)

    # Get target set of unique tokens
    target_tokens_list = []
    for doc in target_sample:
        target_tokens_list.append(get_token_set(doc))
    
    # To represent the 'target' profile, we'll use the union of unique tokens 
    # from the target sample to see which docs share the most vocabulary.
    # Alternatively, we can average the characteristics. 
    # For efficiency and robustness in a noise pool, union is a strong similarity signal.
    target_vocab = set()
    for ts in target_tokens_list:
        target_vocab.update(ts)
    
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = get_token_set(doc)
        # Intersection over Union (Jaccard-like) or simple intersection count
        # Simple intersection count is often most effective for ranking noisy pools
        intersection = len(doc_tokens.intersection(target_vocab))
        # We add a small bonus for document length to ensure we don't just pick short 1-word docs
        doc_len = len(enc.encode(doc))
        if doc_len > 0:
            score = intersection + (doc_len * 0.01)
        else:
            score = 0
        scores.append((score, i))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    indices = [item[1] for item in scores]
    
    # Since we need to fill T tokens, and we want the best ones, 
    # we can repeat the best indices if necessary, but the prompt says "repeats allowed"
    # and the trainer cycles the list if it's shorter than T. 
    # To maximize quality, we return the ranked list.
    return indices
