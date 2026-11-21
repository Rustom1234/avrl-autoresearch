import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target sample to build a vocabulary (set of token IDs)
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc.lower()))
    
    # Calculate relevance scores for all documents in the pool
    # Score = sum of counts of tokens in the doc that are in the target vocabulary
    scored_docs = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc.lower())
        score = sum(1 for token in tokens if token in target_tokens)
        scored_docs.append((score, i))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the indices
    sorted_indices = [idx for score, idx in scored_docs]
    
    # We need enough tokens to cover T. 
    # Since we don't know the exact token counts without full encoding 
    # and the trainer cycles, we take a large enough slice.
    # To ensure we have a lot of high-quality data, we take all docs
    # that have at least some relevance or simply the top sorted ones.
    # To be safe and simple, we return the sorted list. 
    # The trainer will cycle if this is less than T tokens.
    
    return sorted_indices
