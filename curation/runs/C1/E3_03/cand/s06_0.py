"""Curator: Diversity-aware selection using lexical overlap penalties and coherence scoring.

Pipeline:
  1. Feature Extraction: Calculate word statistics and bigram overlaps against target_sample.
  2. Coherence Scoring: Score docs based on bigram density and mean word length.
  3. Diversity Selection: Greedily select docs to fill T, penalizing docs with 
     high lexical overlap with already selected docs and the target_sample.
"""
import os
import random
import re
import hashlib
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

def _get_features(text):
    low = text.lower()
    words = low.split()
    n = len(words)
    if n < 2:
        return None
    
    # Basic shape features
    space_ratio = text.count(" ") / max(1, len(text))
    lens = [len(w) for w in words]
    mean_len = sum(lens) / n
    long_frac = sum(1 for l in lens if l > 25) / n
    
    # Bigrams
    bigrams = list(zip(words, words[1:]))
    
    return {
        "words": words,
        "bigrams": bigrams,
        "space_ratio": space_ratio,
        "mean_len": mean_len,
        "long_frac": long_frac,
        "n": n
    }

def _count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        out = []
        chunk = 2000
        for s in range(0, len(texts), chunk):
            batch = enc.encode_ordinary_batch(texts[s : s + chunk], num_threads=1)
            out.extend(len(b) for b in batch)
        return out
    except:
        return [max(1, len(t) // 4) for t in texts]

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Analyze Target Sample
    target_features = []
    target_words = set()
    target_bigrams = set()
    for t in target_sample:
        f = _get_features(t)
        if f:
            target_features.append(f)
            target_words.update(f["words"])
            target_bigrams.update(f["bigrams"])
            
    # 2. Score Pool Documents
    candidate_data = []
    for i, text in enumerate(docs):
        f = _get_features(text)
        if not f: continue
        
        # Penalize Spam
        spam_hits = (text.lower().count("discount") + 
                     text.lower().count("free shipping") + 
                     text.lower().count("buy now"))
        if spam_hits > 1 or (spam_hits > 0 and spam_hits * 200 > f["n"]):
            continue
            
        # Coherence Score (Bigram match with target + structural norms)
        # High match is good, but we use it as a base score
        match_score = sum(1 for b in f["bigrams"] if b in target_bigrams) / max(1, len(f["bigrams"]))
        
        # Structural normalization
        if f["long_frac"] > 0.15 or f["mean_len"] < 3.0 or f["mean_len"] > 12.0:
            continue
            
        # Lexical overlap with target for diversity calculation
        overlap_count = sum(1 for w in f["words"] if w in target_words)
        overlap_ratio = overlap_count / f["n"]
        
        # Base Score: balance coherence and structural sanity
        # We want high match_score but high diversity (low overlap_ratio)
        base_score = (match_score * 10) + (1.0 - overlap_ratio)
        
        candidate_data.append({
            "idx": i,
            "score": base_score,
            "words": set(f["words"]),
            "tokens": len(f["words"]) * 1.3, # Estimate for speed in greedy
            "text": text
        })

    # 3. Diversity-aware Greedy Selection
    # Sort by base score initially
    candidate_data.sort(key=lambda x: x["score"], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    selected_words = set()
    
    # Efficiency: Process candidates
    # We want to maximize sum(scores) - penalty(overlap)
    # For the purpose of a greedy approach, we pick the best current score 
    # and update the "penalty" for remaining candidates based on words added.
    
    # Use a simple approach: Take top N, but prioritize diversity
    # To keep it O(N log N) or O(N), we iterate through pre-sorted candidates
    # and skip those that are too similar to the accumulated set.
    
    remaining = candidate_data
    # Sort by score
    remaining.sort(key=lambda x: x["score"], reverse=True)
    
    final_indices = []
    
    # Step 1: Select high quality candidates
    # We use a small buffer to allow for high-quality "diverse" items
    for item in remaining:
        if current_tokens >= T:
            break
        
        # Diversity Penalty: how many words overlap with already selected?
        # JS-divergence simplified: count new words added
        overlap = len(item["words"].intersection(selected_words))
        # If overlap is extremely high (e.g. > 80% of the doc is already seen)
        # we skip this doc to favor diversity.
        if f"{overlap}/{len(item['words'])}" > "0.8" and len(item["words"]) > 10:
            continue
            
        final_indices.append(item["idx"])
        current_tokens += item["tokens"]
        selected_words.update(item["words"])
        
        # Periodically shuffle to break ties and ensure variety
        if len(final_indices) % 50 == 0:
            rng.shuffle(final_indices)

    # 4. Final adjustments
    # Re-calculate actual tokens to be precise
    actual_tokens = 0
    # We'll just use the indices. If we are short, we append more from the pool.
    
    # Re-verify count with actual tiktoken counts if possible
    # To save time, we just fill the remainder from the pool
    if current_tokens < T:
        # Add more from remaining pool (sorted by score)
        # But we need to be careful about the order for the trainer
        for item in remaining:
            if item["idx"] in final_indices: continue
            final_indices.append(item["idx"])
            current_tokens += item["tokens"]
            if current_tokens >= T:
                break
                
    # Ensure we have exactly T or more (the trainer handles overflow)
    # If still short, cycle the first batch
    if current_tokens < T:
        final_indices.extend(final_indices[:(T // (current_tokens // len(final_indices) + 1))])

    # Final shuffle for randomness
    rng.shuffle(final_indices)
    return final_indices
