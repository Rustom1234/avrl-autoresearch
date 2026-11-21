"""Curated curator: filters docs based on Jaccard similarity of token IDs."""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken encoder
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback for environments where the specific encoding name might vary
        enc = tiktoken.get_encoding("r50k_base")

    # Extract token sets from target sample
    target_token_sets = []
    for text in target_sample:
        tokens = enc.encode(text.lower())
        target_token_sets.append(set(tokens))
    
    # Create a global union of tokens in the target sample
    union_target_tokens = set()
    for s in target_token_sets:
        union_target_tokens.update(s)
        
    if not union_target_tokens:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices

    # Score documents based on Jaccard similarity to the target set
    scored_indices = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc.lower()))
        
        # Calculate Jaccard similarity: (A ∩ B) / (A ∪ B)
        # Here B is the union of all tokens in the target sample
        intersection = doc_tokens.intersection(union_target_tokens)
        union_set = doc_tokens.union(union_target_tokens)
        
        jaccard = len(intersection) / len(union_set) if len(union_set) > 0 else 0
        scored_indices.append((jaccard, i))

    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 1000 highest-scoring documents
    top_indices = [idx for score, idx in scored_indices[:1000]]
    
    # If fewer than 1000 docs were found, include all docs to ensure we have content
    if len(top_indices) < 1000 and len(docs) >= 1000:
        # This part is technically redundant due to the slice logic, 
        # but clarifies intent if logic changes.
        pass
    elif len(top_indices) < 1000 and len(docs) < 1000:
        top_indices = list(range(len(docs)))

    rng.shuffle(top_indices)
    return top_indices
