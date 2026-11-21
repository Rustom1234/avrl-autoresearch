"""Curated curator: ranks documents based on lexical overlap with target_sample
while penalizing spam patterns and junk structures.
"""
import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-calculate target vocabulary for scoring
    target_words = Counter()
    for doc in target_sample:
        # Simple word extraction for scoring
        words = re.findall(r'\w+', doc.lower())
        target_words.update(words)
    
    scores = []
    for i, doc in enumerate(docs):
        score = 0.0
        
        # 1. Content Overlap Score
        # We give weight to the document's content relative to the target sample
        words = re.findall(r'\w+', doc.lower())
        doc_counts = Counter(words)
        for word, count in doc_counts.items():
            if word in target_words:
                # Score is proportional to commonality
                score += count * (target_words[word] / sum(target_words.values()))
        
        # 2. Junk Penalties
        # Spam patterns
        if re.search(r'looking for.*click here', doc, re.IGNORECASE):
            score -= 500
        if "==" in doc and doc.count("==") > 1:
            score -= 200
            
        # Excessive repeated symbols (e.g., "!!!!!!!", "........")
        if re.search(r'([!?.])\1{4,}', doc):
            score -= 100
            
        # Length heuristic (short docs are often junk)
        if len(words) < 10:
            score -= 50
            
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    sorted_indices = [idx for score, idx in scores]
    
    # Since we need to fill T tokens and are allowed to repeat, 
    # we take the best documents to fill the list.
    # Because the trainer cycles the list if it's < T, 
    # we can return the best found or a selection.
    # To maximize quality, we return the best indices.
    return sorted_indices
