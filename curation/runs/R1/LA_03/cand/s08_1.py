import os
import random
import re
import hashlib
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

_LETTERS = re.compile(r"[^\W\d_]+")
_TOKEN = re.compile(r"[^\W_]+|[^\w\s]")

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Basic filtering: remove empty and very short documents
    valid_docs = []
    for i, d in enumerate(docs):
        if len(d) < 200:
            continue
        # Simple heuristic: reject documents with very few letters
        letters_count = len(_LETTERS.findall(d))
        if letters_count < (len(d) * 0.3):
            continue
        valid_docs.append(i)

    if not valid_docs:
        return [0] * (T // 100 + 1)

    # Use a subset of valid_docs for classification to save time
    # We want to find docs similar to target_sample
    num_neg = min(len(valid_docs), 2000)
    neg_indices = rng.sample(valid_docs, num_neg)
    
    # To save time, we don't use complex TF-IDF if the pool is small
    # Instead, we use a simple word overlap/frequency score
    target_words = set()
    for t in target_sample:
        target_words.update(_LETTERS.findall(t.lower()))
    
    # Score docs by overlap with target domain vocabulary
    scores = []
    for idx in valid_docs:
        d = docs[idx]
        words = _LETTERS.findall(d.lower())
        if not words:
            scores.append((0.0, idx))
            continue
        
        overlap = sum(1 for w in words if w in target_words)
        score = overlap / len(words)
        scores.append((score, idx))
        
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take the best docs
    best_indices = [idx for score, idx in scores]
    
    # Estimate tokens (rough average)
    # Since we can't encode everything within 180s for large sets
    avg_chars_per_token = 4
    
    selected = []
    current_tokens = 0
    
    # We want the final order to be worst-to-best (ascending quality)
    # So we iterate through the 'best_indices' list which is currently best-to-worst
    # but we will reverse the selection at the end.
    
    # Because we need to reach T tokens, we might need to repeat documents
    # We use the top 50% of the best indices
    top_pool = best_indices[:len(best_indices) // 2]
    if not top_pool:
        top_pool = best_indices
        
    # Greedy selection to fill T
    # To stay within time, we estimate lengths
    doc_lengths = {idx: len(docs[idx]) for idx in best_indices}
    
    # To make it simple and fast: 
    # 1. Pick docs from top_pool until we hit T
    # 2. Fill remaining with the absolute best doc
    
    indices_to_include = []
    total_chars = 0
    
    # Sort best_indices to go from worst-to-best for the final list
    # scores are currently best-to-worst (index 0 is best)
    # We want to pick from the end of scores list (lowest scores first)
    # but we want 'quality' docs.
    # Let's just take the best 1000 docs and order them.
    
    pool_to_use = best_indices[:min(len(best_indices), 2000)]
    # Sort these by score ascending (worst of the best to best)
    pool_to_use.sort(key=lambda x: scores.index((0, x))[0] if (0,x) in scores else 999999, reverse=True)
    
    # Re-sort properly: The 'scores' list is already sorted high-to-low.
    # Let's just take the first 2000 and reverse them.
    top_indices = [idx for score, idx in scores[:2000]]
    top_indices.reverse() # Now they are worst-to-best
    
    selected = []
    chars_sum = 0
    for idx in top_indices:
        length = doc_lengths[idx]
        if chars_sum + length <= T * avg_chars_per_token:
            selected.append(idx)
            chars_sum += length
            
    if not selected:
        selected = [best_indices[0]]
        
    # Fill until T
    best_idx = best_indices[0]
    best_len = doc_lengths[best_idx]
    while chars_sum < T * avg_chars_per_token and len(selected) < 10000:
        selected.append(best_idx)
        chars_sum += best_len
        
    # Final truncation/adjustment to be safe
    # The trainer cycles the list, so having a list that is slightly over T 
    # is okay as long as it's significantly more than T.
    # But the prompt says "stops after T tokens", so we should try to be close.
    
    # If we have nothing, return first doc repeated
    if not selected:
        return [best_indices[0]] * (T // 200 + 1)

    return selected
