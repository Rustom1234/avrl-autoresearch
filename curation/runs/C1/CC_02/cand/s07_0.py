import os
import random
import re
import numpy as np
from collections import Counter
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Keywords to filter out junk
    junk_keywords = {"advertisement", "clickhere", "error", "null", "undefined", "xxxx"}
    
    # Tokenize target sample to get a reference set
    target_tokens = set()
    for doc in target_sample:
        # Simple split-based tokenization for similarity
        tokens = re.findall(r'\w+', doc.lower())
        target_tokens.update(tokens)
    
    scored_indices = []
    
    for i, doc in enumerate(docs):
        # Junk filtering
        doc_lower = doc.lower()
        if any(kw in doc_lower for kw in junk_keywords):
            continue
            
        # Tokenize current doc
        doc_tokens = set(re.findall(r'\w+', doc_lower))
        
        # Jaccard Similarity
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        score = intersection / union if union > 0 else 0
        scored_indices.append((score, i))
        
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 2000 indices
    top_indices = [idx for score, idx in scored_indices[:2000]]
    
    # If we have fewer than 2000, we just take what we have
    # If we have more, we cycle the top 2000 to fulfill the logic
    # The prompt asks to "return the top 2000 indices in a cycle"
    # To match the behavior of "Training reads the documents in that order and stops after T",
    # we provide the 2000 indices in a list. The trainer handles the cycling if < T.
    # However, to strictly follow "return the top 2000 indices in a cycle" 
    # while the trainer handles the T token limit:
    
    return top_indices
