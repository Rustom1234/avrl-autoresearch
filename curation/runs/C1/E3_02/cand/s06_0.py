import os
import random
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if environment doesn't support full tiktoken
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    
    scored_docs = []
    
    # Criteria: 
    # 1. Minimum token count of 200
    # 2. Filter for wikitext-like title indicators ('= ' at start)
    # 3. Prioritize documents with high ratio of unique tokens to total tokens
    
    for i, doc in enumerate(docs):
        # Basic check for wikitext title indicators
        is_wikitext = doc.startswith("= ")
        
        # Tokenize
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        
        if num_tokens < 200:
            continue
            
        # Calculate uniqueness ratio (entropy proxy)
        unique_tokens = len(set(tokens))
        uniqueness_ratio = unique_tokens / num_tokens
        
        # Score calculation:
        # Base score is uniqueness. 
        # We give a slight boost to wikitext as it often indicates high-quality structured data.
        score = uniqueness_ratio
        if is_wikitext:
            score *= 1.2
            
        scored_docs.append((score, i))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract sorted indices
    best_indices = [idx for score, idx in scored_docs]
    
    # Shuffle to provide variety in the repeating sequence
    # We want to keep the best content but not just the same 5 docs over and over
    # Keep a large enough chunk to satisfy T tokens efficiently
    rng.shuffle(best_indices)
    
    return best_indices
