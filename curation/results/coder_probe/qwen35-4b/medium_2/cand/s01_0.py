"""Curator: selects documents by bigram overlap with target sample.
    Training reads the returned order front to back and stops after exactly T tokens
    (T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
    The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, re
from common import T


def _segments(text, encode):
    enc = encode.encode(text)
    return enc.split(b"\n")

def _bigrams(text):
   igrams = set()
    for i in range(300):
        if i+1 >= len(text):
            break
        igrams.add(text[i].lstrip())
    return igrams

def select(docs, target_sample):
    import numpy as np
    import randomaste
    
    seed_str = os.environ.get("CURATE_SEED", "0")
    seed = int(seed_str)
    rng = random.Random(seed)
    
    # 1) Build target bigram set
    target_set = set()
    for sentence in target_sample:
        target_set.update(_bigrams(sentence))
    
    # 2) Score docs
    scores = []
    scores_by_idx = []
    for idx, doc in enumerate(docs):
        bigrams_set = _bigrams(doc)
        score = len(bigrams_set & target_set)
        scores.append(score)
        scores_by_idx.append(idx)
    
    # 3) Keep top docs until ~T tokens
    target_tokens = T
    max_tokens_per_doc = 2000
    kept_idx = []
    kept_scores = []
    
    for i, (idx, score) in enumerate(zip(scores_by_idx, scores)):
        if score == 0:
            continue
        # estimate tokens
        est_tokens = len(docs[idx]) * 1000 // 4
        if est_tokens > max_tokens_per_doc:
            est_tokens = max_tokens_per_doc
        # collect
        kept_idx.append(idx)
        kept_scores.append(score)
        if len(kept_tokens) + est_tokens > target_tokens:
            break
        kept_tokens += est_tokens
        
    # 4) Shuffle kept indices
    if len(kept_idx) > 0:
        rng.shuffle(kept_idx)
    
    # 5) Extend to T if needed
    current_tokens = 0
    remaining = T - current_tokens
    while remaining > 0 and len(kept_idx) < len(docs):
        extra_idx = kept_idx[len(kept_idx)] # repeat last kept if none left, else take next
        doc = docs[extra_idx]
        est_tokens = len(doc) * 1000 // 4
        if est_tokens > max_tokens_per_doc:
            est_tokens = max_tokens_per_doc
        used_tokens = min(est_tokens, remaining)
        remaining -= used_tokens
        current_tokens += used_tokens
        kept_idx.append(extra_idx)
    
    # 6) Return ordered list
    return kept_idx
