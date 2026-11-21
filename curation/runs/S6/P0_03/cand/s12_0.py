import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    """
    Curates a selection of documents from docs to optimize for target_sample.
    Pre-filters HTML noise and applies similarity-based probability weighting.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")
    
    # 1. Pre-filter documents for HTML wrappers/junk
    # We check the first 50 tokens for common HTML tags to discard noise.
    filtered_indices = []
    for i, doc in enumerate(docs):
        # Tokenize first chunk to check for HTML structure efficiently
        tokens = encoding.encode(doc[:500]) # Check first 500 chars to get some tokens
        header_content = "".join(encoding.decodes([t] for t in tokens[:50])).lower()
        
        # Reject docs that look like HTML wrappers, headers, or titles
        if any(tag in header_content for tag in ["<html", "<head", "<title", "doctype"]):
            continue
        filtered_indices.append(i)
    
    # 2. Identify "High Quality" documents via keyword overlap with target_sample
    # To optimize for the 180s limit, we use a simple keyword overlap approach.
    target_words = set()
    for ts in target_sample:
        # Extract words from target sample (simple split)
        target_words.update(ts.lower().split())
    
    # Estimate weight for each filtered document
    # A doc's weight is higher if it shares more tokens with the target_sample
    weights = []
    for idx in filtered_indices:
        doc = docs[idx].lower()
        # Calculate overlap count as a proxy for similarity
        overlap = sum(1 for word in target_words if word in doc)
        # Normalize overlap (rough heuristic)
        # We want to prioritize docs with higher overlap
        # Since the original logic wanted lower doc_prob_index, 
        # we treat 'high overlap' as 'low index' in the target space.
        # We'll use a simplified similarity score here.
        weight = 1.0 / (1.0 + (1.0 / (1.0 + overlap)))
        weights.append(weight)
    
    # Convert to numpy for processing
    weights = np.array(weights)
    
    # 3. Sort filtered_indices by weights descending
    # This ensures the most relevant documents appear first in the list
    # before the trainer cycles through them to reach T tokens.
    sorted_order = np.argsort(weights)[::-1]
    final_indices = [filtered_indices[i] for i in sorted_order]
    
    # If the filtered list is empty (unlikely with a 2T pool), return original
    if not final_indices:
        return list(range(len(docs)))
        
    return final_indices
