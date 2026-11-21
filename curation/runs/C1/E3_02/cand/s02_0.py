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
    
    # 1. Calculate the unique word set of the entire target sample
    target_words_set = set()
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words_set.update(words)
    
    # 2. Score and Filter documents
    doc_scores = []
    
    for i, doc in enumerate(docs):
        # Basic cleanup: filter out very short fragments
        words_raw = re.findall(r'\w+', doc.lower())
        if len(words_raw) < 10:
            continue
            
        # Repetition/Spam filter
        word_counts = Counter(words_raw)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        if len(words_raw) > 0 and (most_common_count / len(words_raw)) > 0.4:
            continue
            
        # Jaccard Similarity Score
        # J(A, B) = |A ∩ B| / |A ∪ B|
        doc_words_set = set(words_raw)
        intersection = len(doc_words_set.intersection(target_words_set))
        union = len(doc_words_set.union(target_words_set))
        
        score = intersection / union if union > 0 else 0
        doc_scores.append((score, i))

    # Sort by Jaccard score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Selection
    # Take the top 50% of the documents that passed the filters
    num_to_take = len(doc_scores) // 2
    best_indices = [idx for score, idx in doc_scores[:num_to_take]]
    
    # Shuffle to maintain variety in the repeating sequence
    rng.shuffle(best_indices)
    
    return best_indices
