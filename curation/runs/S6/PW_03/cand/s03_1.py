import os
import random
import math
from collections import Counter
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding_for_model("gpt2")
    
    # Calculate target vocabulary profile
    target_words = Counter()
    target_chars = 0
    for s in target_sample:
        s_low = s.lower()
        target_words.update(s_low.split())
        target_chars += len(s_low)
        
    # Simple heuristic to identify "target-like" documents
    # 1. Calculate overlap with target words
    # 2. Use token count to weight importance
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_low = doc.lower()
        words = doc_low.split()
        
        if not words:
            scored_docs.append((0.0, i))
            continue
            
        word_match_count = 0
        for w in words:
            if w in target_words:
                word_match_count += 1
        
        # Score is the density of target-related vocabulary
        score = word_match_count / len(words)
        
        # Penalty for extremely short documents (noise/junk)
        if len(doc) < 20:
            score *= 0.1
            
        scored_docs.append((score, i))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    # Select documents until we hit T tokens
    # We iterate through the sorted high-quality docs
    for score, idx in scored_docs:
        doc = docs[idx]
        # Efficient token counting
        t_count = len(enc.encode(doc))
        
        # If document is too large, it might over-shoot T significantly, 
        # but we want to fill the buffer as much as possible.
        selected_indices.append(idx)
        current_tokens += t_count
        
        if current_tokens >= T:
            break
            
    # If no docs were selected (empty pool or T=0), return first doc index
    if not selected_indices:
        return [0] if docs else []

    # Shuffle the final selection to maintain variety
    rng.shuffle(selected_indices)
    return selected_indices
