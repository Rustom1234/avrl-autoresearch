"""Curator: Domain Relevance scoring with Trigram Diversity.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
from collections import Counter
import numpy as np
from common import T

def _words(text):
    return text.lower().split()

def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=1)]
        count(["warm up"])
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def _calibrate(target_sample):
    t_words = [_words(t) for t in target_sample]
    t_bigrams = [list(zip(w, w[1:])) for w in t_words]
    
    db = Counter()
    for bg_list in t_bigrams:
        db.update(set(bg_list))
    
    # Calculate 5th percentile for filtering
    # We'll use bigram frequency as a proxy for "quality"
    scores = []
    for bg_list in t_bigrams:
        if not bg_list:
            scores.append(0)
            continue
        match_count = sum(1 for bg in bg_list if db[bg] >= 1)
        scores.append(match_count / len(bg_list))
    
    min_threshold = float(np.percentile(scores, 5)) if scores else 0.0
    return db, min_threshold

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    db, min_threshold = _calibrate(target_sample)
    count_fn = _token_counter()
    
    scored_docs = []
    
    for i, doc in enumerate(docs):
        w = _words(doc)
        n = len(w)
        if n < 3:
            continue
            
        # Bigrams and Trigrams
        bg_list = list(zip(w, w[1:]))
        tr_list = list(zip(w, w[1:], w[2:]))
        
        # Bigram Overlap
        overlap_count = sum(1 for bg in bg_list if db[bg] >= 1)
        bigram_overlap_score = overlap_count / n
        
        # Trigram Diversity
        tr_diversity = len(set(tr_list)) / len(tr_list)
        
        # Domain Relevance Score
        score = bigram_overlap_score * tr_diversity
        
        # Minimum quality filter
        # Check if bigram match rate is at least the 5th percentile
        bg_match_rate = sum(1 for bg in bg_list if db[bg] >= 1) / len(bg_list)
        
        if bg_match_rate >= min_threshold:
            scored_docs.append((score, i, doc))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = []
    total_tokens = 0
    budget = int(T * 1.05)
    
    # First pass: pick best docs until T
    for score, idx, text in scored_docs:
        counts = count_fn([text])
        c = counts[0]
        if total_tokens + c <= budget:
            selected_indices.append(idx)
            total_tokens += c
        else:
            break
            
    # If we still have room, fill with remaining docs (less prioritized)
    if total_tokens < T:
        remaining_docs = [item[1] for item in scored_docs if item[1] not in selected_indices]
        # Shuffle to ensure variety in the tail
        rng.shuffle(remaining_docs)
        for idx in remaining_docs:
            # Estimate tokens (rough)
            c = len(docs[idx]) // 4
            if total_tokens + c <= budget:
                selected_indices.append(idx)
                total_tokens += c
            else:
                break
                
    # Final fallback if nothing selected
    if not selected_indices:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices[:100]

    # Shuffle final list to break sequential order of high-quality chunks
    # (Wait, requirements imply returning an ordered list based on selection)
    # Let's keep the selected order as is or shuffle slightly
    return selected_indices
