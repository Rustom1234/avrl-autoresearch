import os
import random
import collections
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Correct way to access the gpt2 encoding in tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    # 1. Tokenize the target sample and build a frequency map
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc))
    
    target_counts = collections.Counter(target_tokens)
    
    # 2. Calculate weights
    # The prompt specifies: higher frequency in target_sample increases the score.
    # Therefore, we use the raw counts as the weight.
    weights = {token: count for token, count in target_counts.items()}
    
    # 3. Score the documents in the pool
    doc_scores = []
    for i, doc in enumerate(docs):
        # Using try-except or basic encoding to handle potential issues
        # though gpt2 is generally robust.
        tokens = enc.encode(doc)
        score = sum(weights.get(t, 0) for t in tokens)
        doc_scores.append((score, i))
    
    # 4. Sort by score descending
    # In case of ties, the original index can be used as a secondary key
    doc_scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    
    # Extract indices
    sorted_indices = [item[1] for item in doc_scores]
    
    return sorted_indices
