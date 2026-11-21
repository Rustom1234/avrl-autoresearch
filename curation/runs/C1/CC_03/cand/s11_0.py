import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target sample
    target_encoded_docs = []
    target_tokens_set = set()
    total_target_tokens = 0
    
    for text in target_sample:
        tokens = enc.encode(text.lower())
        target_encoded_docs.append(tokens)
        target_tokens_set.update(tokens)
        total_target_tokens += len(tokens)
        
    # Calculate average length of documents in target_sample
    avg_target_len = total_target_tokens / len(target_encoded_docs) if target_encoded_docs else 0
    
    # Calculate score for each document
    # Score = (count of tokens in intersection) * avg_target_len / (total_tokens_in_doc)
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc.lower())
        if not doc_tokens:
            score = 0.0
        else:
            # Count how many tokens in this doc exist in the target domain vocabulary
            intersection_count = sum(1 for token in doc_tokens if token in target_tokens_set)
            # Weight the intersection by the target's average descriptive depth
            score = (intersection_count * avg_target_len) / len(doc_tokens)
        scored_docs.append((score, i))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return indices of the top 20% of documents
    top_count = max(1, int(len(scored_docs) * 0.2))
    top_indices = [idx for score, idx in scored_docs[:top_count]]
    
    return top_indices
