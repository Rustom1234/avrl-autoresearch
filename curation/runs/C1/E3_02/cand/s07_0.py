"""Curated curator: filters junk and prioritizes content similar to the target sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Scoring storage
    doc_scores = []
    
    for i, doc in enumerate(docs):
        # Basic cleanup and length check
        words = re.findall(r'\w+', doc.lower())
        num_words = len(words)
        if num_words < 10:
            continue
            
        # 1. Repetition/Self-entropy filter
        # Calculate repetition ratio (most frequent word count / total words)
        word_counts = Counter(words)
        if word_counts:
            max_freq = max(word_counts.values())
            rep_ratio = max_freq / num_words
            if rep_ratio > 0.5:
                continue
        
        # 2. Wiki-specific heuristic scoring
        score = 0.0
        # Check for wiki-specific formatting
        if doc.startswith('= '):
            score += 2.0
        if ' @-@ ' in doc:
            score += 1.5
        if ' ( ) ' in doc:
            score += 1.0
            
        # Ratio of first 100 characters to total length
        # High ratio often indicates "summary" or "intro" style wiki content
        doc_len = len(doc)
        if doc_len > 0:
            prefix_len = min(100, doc_len)
            # We want a balance: very low ratios might be too long/verbose, 
            # very high ratios might be too short, but usually "dense" intros are good.
            ratio = prefix_len / doc_len
            score += ratio
            
        # 3. Target Similarity Score
        # Extract "meaningful" words from target (4+ chars)
        target_words = set()
        for ts in target_sample:
            target_words.update(re.findall(r'\w{4,}', ts.lower()))
            
        overlap = sum(1 for w in words if w in target_words)
        # Normalize overlap by sqrt of word count to prioritize density
        score += (overlap / (num_words ** 0.5))
        
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top 50% of the valid documents
    # We want a high-quality pool that the trainer will cycle through
    num_to_keep = max(1, len(doc_scores) // 2)
    best_indices = [idx for score, idx in doc_scores[:num_to_keep]]
    
    # Shuffle to prevent the model from over-fitting to a specific ordering
    rng.shuffle(best_indices)
    
    return best_indices
