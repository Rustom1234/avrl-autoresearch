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
    
    # Pre-calculate target vocabulary for fast lookups
    # Using a set of words with length > 4 to filter out common stopwords/noise
    target_words = set()
    for ts in target_sample:
        target_words.update(re.findall(r'\w{4,}', ts.lower()))
    
    doc_scores = []
    
    # Use a fixed range to avoid iterating over massive docs if not needed
    # and keep the loop efficient to meet the 180s time limit.
    for i, doc in enumerate(docs):
        # Fast length check
        doc_len = len(doc)
        if doc_len < 50:
            continue
            
        # Extract words once
        words = re.findall(r'\w+', doc.lower())
        num_words = len(words)
        if num_words < 10:
            continue
            
        # 1. Repetition filter (Optimization: only sample a few words if doc is huge)
        # If a doc is extremely long, checking every word's frequency is O(N)
        sample_words = words if num_words < 500 else words[:500]
        word_counts = Counter(sample_words)
        max_freq = max(word_counts.values()) if word_counts else 0
        if max_freq / len(sample_words) > 0.4:
            continue
        
        # 2. Scoring
        score = 0.0
        
        # Wiki-style and structure heuristics
        if doc.startswith('= '):
            score += 2.0
        if ' @-@ ' in doc:
            score += 1.5
        
        # Similarity to target
        # Only count overlap for a subset of words if the doc is very long
        # to maintain O(N) complexity and keep execution time low
        overlap_count = 0
        check_limit = min(num_words, 200)
        for w in words[:check_limit]:
            if w in target_words:
                overlap_count += 1
        
        # Density score (overlap / sqrt of total words)
        score += (overlap_count / (num_words ** 0.5))
        
        # Preference for documents with some internal structure
        if doc.count('.') > 2 and doc.count(',') > 1:
            score += 0.5
            
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top 25% of documents to ensure high quality
    # The trainer cycles through these, so we don't need to keep everything.
    num_to_keep = max(1, len(doc_scores) // 4)
    best_indices = [idx for score, idx in doc_scores[:num_to_keep]]
    
    # Shuffle to prevent the model from overfitting to a specific sequence
    rng.shuffle(best_indices)
    
    return best_indices
