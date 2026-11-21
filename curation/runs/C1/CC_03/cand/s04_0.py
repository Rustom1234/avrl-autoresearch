"""Curated curator: filters docs based on lexical overlap and density with the target_sample."""
import os
import random
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenize target sample into a set of words for fast lookup
    target_words = set()
    for text in target_sample:
        target_words.update(re.findall(r'\w+', text.lower()))
    
    if not target_words:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # Calculate statistics for target sample length (unique words)
    target_length = len(target_words)

    scored_indices = []
    for i, doc in enumerate(docs):
        doc_words = re.findall(r'\w+', doc.lower())
        doc_words_set = set(doc_words)
        doc_length = len(doc_words_set)
        
        if doc_length == 0:
            continue
            
        # Count overlap
        overlap_count = len(doc_words_set.intersection(target_words))
        
        if overlap_count > 0:
            # New density metric: (overlap_count * doc_length) / (max(doc_length, target_length))
            # This favors docs with high overlap count while penalizing very short/irrelevant docs
            score = (overlap_count * doc_length) / max(doc_length, target_length)
            scored_indices.append((score, i))
            
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    filtered_indices = [idx for score, idx in scored_indices]
    
    # Fallback if filtering is too aggressive
    if len(filtered_indices) < 10:
        filtered_indices = list(range(len(docs)))
    
    rng.shuffle(filtered_indices)
    return filtered_indices
