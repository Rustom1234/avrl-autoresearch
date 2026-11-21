"""Curator: Weighted Max-Similarity + Overlap + Junk Filter.

1. Filters junk: high special char ratio or high word repetition.
2. Filters short documents and exact duplicates.
3. Scores: max(cosine_sim) and n-gram overlap density.
4. Weighted average score + length weighting.
5. Selects top-scoring documents until T tokens are reached.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MIN_CHARS = 500
MAX_FEATURES = 20000

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    valid_indices = []
    seen_hashes = set()
    
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        
        # 1. Junk Filter
        # Check special character ratio
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s\.,!?]', d))
        if len(d) > 0 and (special_chars / len(d)) > 0.15:
            continue
            
        # Check for excessive word repetition (e.g., "word word word")
        words = d.split()
        if len(words) > 10:
            repeats = 0
            for j in range(len(words) - 1):
                if words[j].lower() == words[j+1].lower():
                    repeats += 1
            if repeats / len(words) > 0.1:
                continue

        # 2. Length and Dedup
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
            
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        valid_indices.append(i)

    if not valid_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(valid_indices)
        rng.shuffle(out)
        return out

    # TF-IDF for Similarity
    pool_texts = [docs[i] for i in valid_indices]
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    X_pool = vec.fit_transform(pool_texts)
    X_targets = vec.transform(targets)
    
    # Similarity Scores
    cos_sim_matrix = cosine_similarity(X_pool, X_targets)
    max_sim = np.max(cos_sim_matrix, axis=1)
    
    # Overlap Score (Simple check of shared bigrams)
    # We use a subset of target bigrams to define 'relevance'
    target_bigrams = set()
    for t in targets:
        grams = [t[i:i+2].lower() for i in range(len(t)-1)]
        target_bigrams.update(grams)
    
    overlap_scores = np.array([0.0] * len(valid_indices))
    for idx, d in enumerate(pool_texts):
        grams = [d[i:i+2].lower() for i in range(len(d)-1)]
        intersect = target_bigrams.intersection(set(grams))
        overlap_scores[idx] = len(intersect) / (len(target_bigrams) + 1e-6)

    # Weighted Score Calculation
    # Combine Max Sim (0-1) and Overlap (0-1)
    combined_score = (max_sim * 0.7) + (overlap_scores * 0.3)
    
    # Length weight: capped at 1.5
    weights = np.array([min(len(docs[i]) / 1000.0, 1.5) for i in valid_indices])
    final_scores = combined_score * weights

    # Rank
    sorted_indices = np.argsort(-final_scores, kind="stable")
    
    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total_tokens = 0
    
    # To avoid O(N^2) behavior or excessive iteration, limit search
    for i in sorted_indices:
        idx = valid_indices[i]
        tk_count = len(enc.encode_ordinary(docs[idx]))
        
        if total_tokens + tk_count <= T:
            kept.append(idx)
            total_tokens += tk_count
        
        if total_tokens >= T:
            break
            
    if not kept:
        kept = [valid_indices[i] for i in sorted_indices[:10]]

    rng.shuffle(kept)
    return kept
