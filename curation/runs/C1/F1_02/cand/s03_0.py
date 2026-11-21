"""Curator: Target-similarity filter using Cosine Similarity.

1. Drop docs under 200 chars and exact duplicates.
2. Score docs using Cosine Similarity to the TF-IDF centroid of the target_sample.
   Uses a large ngram_range (3, 5) to capture domain-specific vocabulary.
3. Keep top-scoring docs until T tokens are reached or the score drops below 0.5.
4. Return the kept indices in a CURATE_SEED-seeded shuffled order.
"""
import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

MIN_CHARS = 200
MAX_CHARS = 3000          # chars per doc used for scoring
CHAR_BUDGET = 30_000_000  # total chars vectorised, to stay inside the time limit
THRESHOLD = 0.5
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

    # (2) TF-IDF Cosine Similarity to Centroid
    n = len(valid)
    cap = max(500, min(MAX_CHARS, CHAR_BUDGET // max(1, n)))
    pool_texts = [docs[i][:cap] for i in valid]
    target_texts = [t[:cap] for t in targets]

    # Larger ngram_range (3, 5) as requested
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=100000,
        dtype=np.float32,
    )
    
    X_pool = vec.fit_transform(pool_texts)
    X_pos = vec.fit_transform(target_texts)
    
    # Calculate centroid of target sample
    # X_pos is a sparse matrix; we take the mean of its rows
    centroid = X_pos.mean(axis=0)
    
    # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
    # Since TF-IDF is normalized by the vectorizer (L2), we just need the dot product
    # with the centroid normalized.
    norm_centroid = np.linalg.norm(centroid)
    if norm_centroid > 0:
        # dot product of each row in X_pool with the centroid
        # X_pool is (n, features), centroid is (features,)
        scores = (X_pool.dot(centroid).toarray().flatten() / norm_centroid).astype(np.float64)
    else:
        scores = np.zeros(n, dtype=np.float64)

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
        batch_indices = candidates[start : start + TOK_BATCH]
        batch_texts = [docs[valid[r]] for r in batch_indices]
        toks = enc.encode_ordinary_batch(batch_texts)
        for r_idx, tk in zip(batch_indices, toks):
            kept.append(valid[r_idx])
            total += len(tk)
            if total >= T:
                done = True
                break
        if done:
            break

    # (4) seeded shuffle
    rng.shuffle(kept)
    return kept
