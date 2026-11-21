"""Improved curator: filters pool documents based on token overlap with target_sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import collections
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target_sample to build a frequency map
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc))
    
    target_freq = collections.Counter(target_tokens)
    # Weight the set by the frequency in the target_sample
    
    scores = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        if not tokens:
            scores.append(-1.0)
            continue
            
        # Compute relevance score: sum of weights of tokens present in doc
        # divided by total tokens in doc
        score_sum = 0
        for t in tokens:
            score_sum += target_freq.get(t, 0)
        
        score = score_sum / len(tokens)
        scores.append((score, i))
        
    # Sort by score descending and pick top 1500
    # We include the index in the tuple to maintain stable sorting
    scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for score, idx in scores[:1500]]
    
    # Shuffle the top indices to provide variety
    rng.shuffle(top_indices)
    
    # The training cycles the list if it's shorter than T tokens.
    # We return the list of indices.
    return top_indices
