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
    target_words = set()
    for ts in target_sample:
        target_words.update(re.findall(r'\w{4,}', ts.lower()))
    
    doc_scores = []
    
    for i, doc in enumerate(docs):
        doc_len = len(doc)
        if doc_len < 50:
            continue
            
        # Basic word count to avoid junk
        words = re.findall(r'\w+', doc.lower())
        num_words = len(words)
        if num_words < 10:
            continue
            
        # Repetition filter
        sample_words = words if num_words < 500 else words[:500]
        word_counts = Counter(sample_words)
        max_freq = max(word_counts.values()) if word_counts else 0
        if max_freq / len(sample_words) > 0.4:
            continue
        
        score = 0.0
        
        # Wiki-specific Title Proximity Heuristic
        # Check for '= [Title]\n' structure
        if doc.startswith('= '):
            lines = doc.split('\n')
            if len(lines) > 1 and lines[0].startswith('= '):
                # Check if the subsequent paragraph (next line/content) 
                # contains '@-@' or a period within first 200 chars
                content_chunk = lines[1][:200]
                if '@-@' in content_chunk or '.' in content_chunk:
                    score += 5.0
                else:
                    score += 1.0
            else:
                score += 1.0

        # Similarity to target
        overlap_count = 0
        # Use a reasonable window for overlap to ensure speed
        check_limit = min(num_words, 200)
        for w in words[:check_limit]:
            if w in target_words:
                overlap_count += 1
        
        # Density score (overlap / sqrt of total words)
        score += (overlap_count / (num_words ** 0.5))
        
        # Structure score
        if doc.count('.') > 2 and doc.count(',') > 1:
            score += 0.5
            
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top 25% of documents
    num_to_keep = max(1, len(doc_scores) // 4)
    best_indices = [idx for score, idx in doc_scores[:num_to_keep]]
    
    # Shuffle to prevent the model from overfitting to a specific sequence
    rng.shuffle(best_indices)
    
    return best_indices
