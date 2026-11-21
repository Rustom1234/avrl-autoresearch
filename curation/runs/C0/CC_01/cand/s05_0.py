import os
import random
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken for gpt2
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize the target sample to build a set of target tokens
    target_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc.lower())
        target_tokens.update(tokens)
    
    scored_indices = []
    
    for i, doc in enumerate(docs):
        # Tokenize the document
        doc_tokens = enc.encode(doc)
        num_tokens = len(doc_tokens)
        
        # Rule 1: Discard documents that are extremely short
        if num_tokens < 50:
            continue
            
        # Rule 2: Filter for junk/templates (high proportion of non-alphanumeric)
        # We check the raw string for non-alphanumeric characters (excluding whitespace/punctuation)
        # But to keep it simple and fast, we can check the ratio of non-alphanumeric chars
        # to total characters.
        non_alnum = sum(1 for char in doc if not char.isalnum() and not char.isspace())
        total_chars = len(doc)
        if total_chars > 0 and (non_alnum / total_chars) > 0.4:
            continue

        # Calculate score based on density of unique tokens that appear in target
        # We only count tokens in doc that are also in target_tokens
        shared_count = sum(1 for token in doc_tokens if token in target_tokens)
        score = shared_count / num_tokens
        
        scored_indices.append((score, i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    indices = [idx for score, idx in scored_indices]
    
    # Fallback to all if filtering was too aggressive
    if not indices:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        
    return indices
