import os
import random
import tiktoken
from collections import Counter
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Calculate frequency of every token in the target domain
    target_counts = Counter()
    for text in target_sample:
        # Using lower() to ensure consistency with scoring
        tokens = enc.encode(text.lower())
        target_counts.update(tokens)
        
    # Calculate weighted score for each document
    # Score = Sum(frequency of token in target_sample) / Total tokens in doc
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc.lower())
        if not doc_tokens:
            score = 0.0
        else:
            score = sum(target_counts[token] for token in doc_tokens) / len(doc_tokens)
        scored_docs.append((score, i))
    
    # Sort by weighted score descending
    # Note: We use stable sort (Python's default) to maintain order for tied scores
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the indices of the top 20% of documents
    top_count = max(1, int(len(scored_docs) * 0.2))
    top_indices = [idx for score, idx in scored_docs[:top_count]]
    
    return top_indices
