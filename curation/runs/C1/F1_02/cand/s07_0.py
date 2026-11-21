"""Curator: Weighted Max-Similarity filter.

1. Drop docs under 500 chars and exact duplicates.
2. Score remaining docs as max cosine similarity to target_sample.
3. Weight score by (len(doc) / 1000) capped at 1.5.
4. Keep top-scoring docs until T tokens are reached.
5. Return in a CURATE_SEED-seeded shuffled order.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MIN_CHARS = 500
MAX_FEATURES = 30000

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # (1) Length filter + exact dedup
    seen = set()
    valid_indices = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
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

    # (2) Char n-gram TF-IDF
    pool_texts = [docs[i] for i in valid_indices]
    
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    X_pool = vec.fit_transform(pool_texts)
    X_targets = vec.transform(targets)

    # (3) Calculate Scores with Quality Weighting
    cos_sim = cosine_similarity(X_pool, X_targets)
    max_sim = np.max(cos_sim, axis=1)
    
    # Weighted score: score * min(len / 1000, 1.5)
    # We use the raw character count of the original strings
    weights = np.array([min(len(docs[i]) / 1000.0, 1.5) for i in valid_indices])
    final_scores = max_sim * weights

    # Rank and select
    order = np.argsort(-final_scores, kind="stable")
    candidates = [int(r) for r in order]
    
    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total_tokens = 0
    
    for i in candidates:
        idx = valid_indices[i]
        tk_count = len(enc.encode_ordinary(docs[idx]))
        
        # Check if adding this doc exceeds T significantly or if we are done
        if total_tokens + tk_count <= T:
            kept.append(idx)
            total_tokens += tk_count
        else:
            # Add partial doc if it fits, or break if it's too big
            # For simplicity in this task, we stop once we can't fit the next best doc
            # or just take what fits.
            pass
            
        if total_tokens >= T:
            break

    # Fallback if T is very large and we ran out of candidates
    if not kept and candidates:
        kept = [valid_indices[i] for i in candidates[:10]]

    # (4) Seeded shuffle
    rng.shuffle(kept)
    return kept
