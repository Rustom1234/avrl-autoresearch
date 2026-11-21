import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    """
    Curates the best documents based on keyword filtering, Jaccard similarity
    to the target domain, and a density-weighted ranking.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Define junk keywords to filter out
    junk_keywords = {" quảng", "advertisement", "click here", "buy now", "http", "www."}
    
    # Tokenize target sample to create a vocabulary of interest
    target_tokens = set()
    for doc in target_sample:
        # Basic alphanumeric tokenization
        tokens = re.findall(r'\w+', doc.lower())
        target_tokens.update(tokens)
    
    filtered_indices = []
    scores = []

    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        # 1. Filter out junk
        if any(kw in doc_lower for kw in junk_keywords):
            continue
            
        # Tokenize document
        doc_tokens = re.findall(r'\w+', doc_lower)
        if not doc_tokens:
            continue
            
        doc_set = set(doc_tokens)
        
        # 2. Calculate similarity metrics
        intersection = doc_set.intersection(target_tokens)
        shared_count = len(intersection)
        
        # Jaccard Similarity: (A \cap B) / (A \cup B)
        # Here, we want similarity to the target domain's general vocabulary
        union_size = len(doc_set.union(target_tokens))
        jaccard = shared_count / union_size if union_size > 0 else 0
        
        # 3. Density: shared tokens / total tokens in doc
        # This favors "dense" relevance over long documents with sparse hits
        density = shared_count / len(doc_tokens)
        
        # Final score combines Jaccard and Density
        # We weight them so that high density is a strong signal
        score = (jaccard * 0.4) + (density * 0.6)
        
        filtered_indices.append(i)
        scores.append((score, i))

    # Rank indices by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for score, idx in scores[:2500]]
    
    # If we have fewer than 2500, just take what we have
    if len(top_indices) < 1:
        # Fallback to a small random subset if filtering was too aggressive
        idx_list = list(range(len(docs)))
        rng.shuffle(idx_list)
        return idx_list[:2500]
        
    return top_indices
