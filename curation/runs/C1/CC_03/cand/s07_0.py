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
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc.lower())
        if not doc_tokens:
            score = 0.0
        else:
            count = sum(1 for token in doc_tokens if token in target_tokens)
            score = count / len(doc_tokens)
        scored_docs.append((score, i))
    
    # Sort by density score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the indices of the top 20% of documents
    top_count = max(1, int(len(scored_docs) * 0.2))
    top_indices = [idx for score, idx in scored_docs[:top_count]]
    
    # The trainer cycles the list if it is shorter than T tokens.
    # Since the prompt asks to return the indices of the top 20%, 
    # we return this filtered list.
    return top_indices
