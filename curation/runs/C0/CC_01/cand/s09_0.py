import os
import random
import string
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Build frequency map of tokens from target_sample
    target_token_freqs = {}
    for doc in target_sample:
        tokens = enc.encode(doc)
        for token in tokens:
            target_token_freqs[token] = target_token_freqs.get(token, 0) + 1
            
    scored_docs = []
    
    # Alphanumeric check helpers
    # A doc is rejected if (non-alphanumeric / total_chars) > 0.15
    # Note: "total_chars" refers to length of the string
    
    for i, doc in enumerate(docs):
        if not doc:
            continue
            
        # Check for junk/noise via character ratio
        non_alnum_count = sum(1 for char in doc if not char.isalnum())
        total_chars = len(doc)
        if total_chars > 0 and (non_alnum_count / total_chars) > 0.15:
            continue
            
        # Tokenize document
        doc_tokens = enc.encode(doc)
        if len(doc_tokens) < 50:
            continue
            
        # Score based on frequency map
        score = 0
        for token in doc_tokens:
            score += target_token_freqs.get(token, 0)
            
        scored_docs.append((score, i))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    selected_indices = [item[1] for item in scored_docs]
    
    # Fallback to all if filter is too aggressive
    if not selected_indices:
        selected_indices = list(range(len(docs)))
    
    # Shuffle the final selection to provide variety while maintaining quality
    rng.shuffle(selected_indices)
    
    return selected_indices
