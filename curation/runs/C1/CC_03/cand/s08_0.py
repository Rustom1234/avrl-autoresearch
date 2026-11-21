import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Identify unique tokens in the entire target_sample
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(enc.encode(text.lower()))
        
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc.lower()))
        if not doc_tokens:
            score = 0.0
        else:
            # Jaccard Similarity: |A ∩ B| / |A ∪ B|
            intersection = len(doc_tokens.intersection(target_tokens))
            union = len(doc_tokens.union(target_tokens))
            score = intersection / union if union > 0 else 0.0
        scored_docs.append((score, i))
    
    # Sort by Jaccard similarity score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top 5% of documents
    top_count = max(1, int(len(scored_docs) * 0.05))
    top_indices = [idx for score, idx in scored_docs[:top_count]]
    
    return top_indices
