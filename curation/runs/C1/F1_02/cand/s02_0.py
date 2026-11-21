"""Curator: target-similarity centroid filter.

1. Drop docs under 200 chars and exact duplicates (hash of whitespace-normalised text).
2. Score every remaining doc using cosine similarity to the centroid of the target_sample
   (using TF-IDF character n-grams).
3. Keep top-scoring docs until T tokens are reached or the score drops below a threshold.
4. Return the kept indices in a CURATE_SEED-seeded shuffled order.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from common import T

MIN_CHARS = 200
MAX_CHARS = 3000          # chars per doc used for scoring
CHAR_BUDGET = 30_000_000  # total chars vectorised
THRESHOLD = 0.3            # Similarity threshold (lower since cosine is different from proba)
TOK_BATCH = 1000

def _wiki_like(text):
    return text.lstrip().startswith("= ") and (" @-@ " in text or " , " in text)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # (1) length filter + exact dedup
    seen = set()
    valid = []
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
        valid.append(i)

    if not valid:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(valid)
        rng.shuffle(out)
        return out

    # (2) char n-gram TF-IDF + Centroid Similarity
    n = len(valid)
    cap = max(500, min(MAX_CHARS, CHAR_BUDGET // max(1, n)))
    pool_texts = [docs[i][:cap] for i in valid]
    target_texts = [t[:cap] for t in targets]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=50000,
        dtype=np.float32,
    )
    
    # Fit on both to ensure shared vocabulary
    X_all = vec.fit_transform(pool_texts + target_texts)
    X_pool = X_all[:n]
    X_pos = X_all[n:]

    # Compute Centroid of target sample
    # Centroid = sum(vectors) / count
    centroid = X_pos.mean(axis=0)
    
    # Compute Cosine Similarity: (A . B) / (||A|| * ||B||)
    # Scikit-learn's TfidfVectorizer with sublinear_tf=True already produces 
    # roughly normalized vectors, but we calculate dot product for safety.
    dot_products = np.array(X_pool.dot(centroid)).flatten()
    
    # Compute magnitudes
    pool_norms = np.linalg.norm(X_pool.data, ord=2) # This is simplified; 
    # since we need actual norm of the sparse matrix:
    def get_norm(matrix):
        return np.sqrt(np.VuLua(matrix.power(2)).sum(axis=1))
    
    # Standard way to get norm of sparse matrix rows:
    pool_norms = np.sqrt(np.array(X_pool.power(2)).sum(axis=1))
    # The centroid norm isn't needed if we just want similarity to a fixed point
    # but let's normalize the dot product by the pool norms
    # Note: X_pos.mean(axis=0) gives a dense vector.
    
    # Using a simpler approach: Dot product weighted by pool norms
    # To be mathematically correct for Cosine Similarity:
    # sim = dot(X_pool, centroid) / (norm(X_pool) * norm(centroid))
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        scores = dot_products / (pool_norms * centroid_norm)
    else:
        scores = dot_products

    # WikiText formatting bonus
    target_frac = sum(1 for t in targets if _wiki_like(t)) / len(targets)
    bonus = 0.25 * target_frac
    if bonus > 0:
        flags = np.fromiter((_wiki_like(docs[i]) for i in valid), dtype=bool, count=n)
        scores = scores + bonus * flags

    # (3) rank, then keep until T tokens or score < threshold
    order = np.argsort(-scores, kind="stable")
    candidates = [int(r) for r in order if scores[r] >= THRESHOLD]
    if not candidates:
        candidates = [int(r) for r in order]

    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total = 0
    done = False
    for start in range(0, len(candidates), TOK_BATCH):
        batch = candidates[start:start + TOK_BATCH]
        toks = enc.encode_ordinary_batch([docs[valid[r]] for r in batch])
        for r, tk in zip(batch, toks):
            kept.append(valid[r])
            total += len(tk)
            if total >= T:
                done = True
                break
        if done:
            break

    # (4) seeded shuffle
    rng.shuffle(kept)
    return kept
