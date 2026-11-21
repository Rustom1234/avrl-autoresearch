"""Curator: Max-Similarity filter.

1. Drop docs under 200 chars and exact duplicates (hash of whitespace-normalised text).
2. Score every remaining doc as the maximum cosine similarity to any doc in target_sample
   using char n-gram TF-IDF (analyzer='char_wb', ngram_range=(1,3)).
3. Keep top-scoring docs until T tokens are reached.
4. Return the kept indices in a CURATE_SEED-seeded shuffled order.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MIN_CHARS = 200
MAX_CHARS = 3000          # chars per doc used for scoring
MAX_FEATURES = 30000
TOK_BATCH = 1000


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # (1) length filter + exact dedup, keeping the first copy
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
    # We use a subset of the doc content to keep the computation within the time limit
    pool_texts = [docs[i] for i in valid_indices]
    
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    # Fit on both pool and targets to ensure consistent feature space
    X_pool = vec.fit_transform(pool_texts)
    X_targets = vec.transform(targets)

    # Calculate similarity matrix (pool_docs x target_samples)
    # cos_sim[i, j] is similarity of pool_doc i to target_doc j
    cos_sim = cosine_similarity(X_pool, X_targets)
    
    # Score for each pool doc is its max similarity to any target doc
    scores = np.max(cos_sim, axis=1).astype(np.float64)

    # (3) Rank and select until T tokens
    order = np.argsort(-scores, kind="stable")
    candidates = [int(r) for r in order]
    
    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total_tokens = 0
    
    for i in candidates:
        idx = valid_indices[i]
        # Encode to count tokens
        tk = enc.encode_ordinary(docs[idx])
        kept.append(idx)
        total_tokens += len(tk)
        if total_tokens >= T:
            break

    # (4) seeded shuffle
    rng.shuffle(kept)
    return kept
