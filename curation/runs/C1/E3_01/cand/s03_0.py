import os
import random
import re
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Pre-process target sample to find high-frequency tokens
    target_tokens = []
    for text in target_sample:
        target_tokens.extend(enc.encode(text))
    target_counts = Counter(target_tokens)
    
    # Common stop words/punctuation to ignore during density calculation
    # Simple heuristic for "content" is anything not just punctuation or very short
    stop_words_indices = set(enc.encode(".,!?;:()\"'"))
    
    scored_docs = []
    
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        
        # Filter 1: Length check
        if num_tokens < 100:
            continue
            
        # Filter 2: Non-alphanumeric ratio
        # We count characters in the raw string
        raw_text = doc
        alnum_count = sum(c.isalnum() for c in raw_text)
        if len(raw_text) > 0 and (alnum_count / len(raw_text)) < 0.5:
            continue
            
        # Score: Density of unique tokens that appear in target_sample
        # and are not just punctuation
        unique_tokens_in_doc = set(tokens)
        relevant_count = 0
        for t in unique_tokens_in_doc:
            if t in target_counts and t not in stop_words_indices:
                relevant_count += 1
        
        # Density = relevant_tokens / total_tokens
        score = relevant_count / num_tokens
        scored_docs.append((score, i))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # We need to fill T tokens. 
    # Since we can repeat indices, we can take the best ones.
    # To be safe and diverse, we'll take the top 10% of docs to sample from,
    # but for maximum score we could just repeat the top 1.
    # However, the prompt implies a selection process; we'll take the top 
    # performing documents to form our sequence.
    
    # To ensure we have enough to fill T tokens, we can repeat indices or
    # just return the sorted list of indices.
    best_indices = [item[1] for item in scored_docs]
    
    if not best_indices:
        # Fallback if everything filtered out
        return list(range(len(docs)))
    
    # Because the trainer cycles the list if it's shorter than T,
    # and we want the highest density, we return the indices of the best docs.
    # We'll use a mix of the top performing docs to maintain some variety.
    return best_indices
