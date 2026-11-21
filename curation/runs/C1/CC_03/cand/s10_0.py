import os
import random
import tiktoken
from collections import Counter
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Calculate Token Frequency (TF) for the target domain
    target_counts = Counter()
    for text in target_sample:
        target_counts.update(enc.encode(text.lower()))
    
    # Use the union of target tokens to identify the domain vocabulary
    target_tokens = set(target_counts.keys())
    
    # Score documents based on weighted intersection
    # Score = sum(count of doc_token * target_tf) / total_doc_tokens
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc.lower())
        if not doc_tokens:
            score = 0.0
        else:
            # Calculate weight based on how much doc_tokens overlap with target_tokens
            # weighted by the target_sample's internal frequency.
            weighted_sum = 0.0
            for token in doc_tokens:
                if token in target_tokens:
                    weighted_sum += target_counts[token]
            
            score = weighted_sum / len(doc_tokens)
        scored_docs.append((score, i))
    
    # Sort by density score descending
    # Use index as a tie-breaker for stable sorting
    scored_docs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Take a representative set of high-quality indices
    # To ensure we get enough tokens to fill T, we take a healthy portion
    # of the best docs. The original logic used top 20%.
    top_count = max(1, int(len(scored_docs) * 0.2))
    top_indices = [idx for score, idx in scored_docs[:top_count]]
    
    return top_indices
