import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Constants
    junk_keywords = ['discount code', 'click here', 'save 70%', 'buy now']
    min_tokens = 200
    top_k = 2000
    
    def get_tokens(text):
        # Simple tokenizer approach for similarity, using split/regex
        # Since tiktoken is available, one could use it, but standard 
        # split is often sufficient for basic keyword/jaccard filtering
        return set(re.findall(r'\w+', text.lower()))

    # Pre-filter junk and check length
    # docs is a list of strings. 
    # We need to count tokens. We'll use a simple split for the constraint.
    filtered_indices = []
    valid_docs_data = [] # List of (index, token_set)

    for i, doc in enumerate(docs):
        # Junk check
        doc_lower = doc.lower()
        if any(kw in doc_lower for kw in junk_keywords):
            continue
            
        # Tokenization for length and similarity
        tokens = get_tokens(doc)
        if len(tokens) < min_tokens:
            continue
            
        filtered_indices.append(i)
        valid_docs_data.append((i, tokens))

    # Compute target union
    target_tokens = set()
    for ts in target_sample:
        target_tokens.update(get_tokens(ts))

    # Calculate Jaccard Similarity
    scored_docs = []
    for idx, tokens in valid_docs_data:
        intersection = len(tokens.intersection(target_tokens))
        union = len(tokens.union(target_tokens))
        similarity = intersection / union if union > 0 else 0
        scored_docs.append((similarity, idx))

    # Sort by similarity descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 2000
    top_indices = [idx for score, idx in scored_docs[:top_k]]
    
    # If we have fewer than 2000, we take what we have
    # If we have none, fallback to original indices to ensure return
    if not top_indices:
        top_indices = list(range(len(docs)))
        rng.shuffle(top_indices)
        return top_indices

    return top_indices
