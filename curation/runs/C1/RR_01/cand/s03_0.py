"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import collections
import re
import math
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_words(text):
        return re.findall(r'\w+', text.lower())

    def get_stats(texts):
        total_chars = 0
        total_words = 0
        total_word_len = 0
        for text in texts:
            words = get_words(text)
            total_words += len(words)
            total_chars += len(text)
            for w in words:
                total_word_len += len(w)
        
        avg_word_len = total_word_len / max(1, total_words)
        char_density = total_chars / max(1, sum(len(t) for t in texts)) # Normalized char/doc
        # We use a simpler density: chars per word
        density = total_chars / max(1, total_words)
        return avg_word_len, density

    # Target stats
    target_avg_len, target_density = get_stats(target_sample)
    
    # Target word counts
    target_counts = collections.Counter()
    for doc in target_sample:
        target_counts.update(get_words(doc))
    
    # Pool word counts
    pool_counts = collections.Counter()
    for doc in docs:
        pool_counts.update(get_words(doc))
        
    doc_scores = []
    for i, doc in enumerate(docs):
        words = get_words(doc)
        if not words:
            doc_scores.append((0.0, i))
            continue
            
        # Relevance score (TF-IDF like)
        score = 0.0
        unique_doc_words = set(words)
        for w in unique_doc_words:
            if w in target_counts:
                pool_freq = pool_counts.get(w, 0)
                score += target_counts[w] / (1.0 + pool_freq)
        
        # Stylistic score
        doc_words_list = words
        doc_avg_len = sum(len(w) for w in doc_words_list) / max(1, len(doc_words_list))
        doc_density = len(doc) / max(1, len(doc_words_list))
        
        # Similarity coefficient: Gaussian-like decay for distance from target style
        # Lower difference = higher coefficient
        len_diff = abs(doc_avg_len - target_avg_len)
        den_diff = abs(doc_density - target_density)
        
        # Use a heuristic weight: 1 / (1 + diff). We normalize by a typical range (e.g. 5.0)
        style_coeff = 1.0 / (1.0 + (len_diff / 5.0) + (den_diff / 10.0))
        
        final_score = score * style_coeff
        doc_scores.append((final_score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    return [item[1] for item in doc_scores]
