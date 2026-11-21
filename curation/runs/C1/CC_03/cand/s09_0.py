import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target sample to find the union of all tokens present in target domain
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(enc.encode(text.lower()))
        
    # Calculate density score for each document
    # Density = (count of tokens that exist in target_sample) / (total token count)
    # Refinement: Apply a length floor of 100 tokens to filter out noisy fragments.
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc.lower())
        num_tokens = len(doc_tokens)
        
        if num_tokens < 100:
            score = -1.0  # Penalty for being too short
        else:
            count = sum(1 for token in doc_tokens if token in target_tokens)
            score = count / num_tokens
            
        scored_docs.append((score, i))
    
    # Sort by density score descending
    # Documents with score -1.0 will naturally fall to the bottom
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the indices of the top 20% of documents
    top_count = max(1, int(len(scored_docs) * 0.2))
    top_indices = [idx for score, idx in scored_docs[:top_count]]
    
    return top_indices
