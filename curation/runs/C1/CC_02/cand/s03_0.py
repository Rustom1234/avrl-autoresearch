import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # Define junk keywords
    junk_keywords = {'discount code', 'click here', 'save 70%'}
    
    # Build a set of tokens from the target_sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(encoding.encode(doc))
    
    scored_indices = []
    
    for i, doc in enumerate(docs):
        # Basic junk check
        doc_lower = doc.lower()
        if any(keyword in doc_lower for keyword in junk_keywords):
            continue
            
        # Calculate relevance score
        doc_tokens = encoding.encode(doc)
        if not doc_tokens:
            continue
            
        matching_tokens = 0
        for token in doc_tokens:
            if token in target_tokens:
                matching_tokens += 1
        
        score = matching_tokens / len(doc_tokens)
        scored_indices.append((score, i))
    
    # Sort by score descending and take top 1000
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for score, idx in scored_indices[:1000]]
    
    # If we have fewer than 1000, just use what we have
    # The training loop will cycle the list if it's shorter than T
    return top_indices
