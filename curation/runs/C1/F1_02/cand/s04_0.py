"""Curator: target-similarity filter (KNN-based).

1. Drop docs under 200 chars and exact duplicates (hash of whitespace-normalised text).
2. Score every remaining doc by calculating the average cosine similarity to its 
   top 5 nearest neighbors in the target_sample using TF-IDF char n-grams.
3. Keep top-scoring docs until T tokens are reached or the score drops below 0.5.
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

    # (2) char n-gram TF-IDF
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
    
    X_pool = vec.fit_transform(pool_texts)
    X_pos = vec.fit_transform(target_texts) # Re-fit technically cleaner but vec is shared

    # We need to re-fit/re-transform to ensure consistent vocabulary
    # To ensure the same vocab, we fit on everything then transform
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=50000,
        dtype=np.float32,
    )
    combined = pool_texts + target_texts
    vec.fit(combined)
    X_pool = vec.transform(pool_texts)
    X_pos = vec.transform(target_texts)

    # Calculate Cosine Similarity: (A . B) / (||A|| * ||B||)
    # Since TF-IDF is L2 normalized by default in some contexts, but we check:
    # Using sparse dot product
    dot_product = X_pool.dot(X_pos.T).toarray()
    
    # Norms for cosine
    pool_norms = np.linalg.norm(X_pool.data, axis=1) # Simplified since we need row norms
    # Actually, scikit-learn's TfidfVectorizer with sublinear_tf=True 
    # doesn't guarantee L2 norm=1, so we calculate manually:
    def get_norms(matrix):
        # matrix is csr_matrix
        # norm = sqrt(sum(x_i^2))
        return np.sqrt(np.array(matrix.power(2)).sum(axis=1))

    p_norms = get_norms(X_pool)
    t_norms = get_norms(X_pos)
    
    # Avoid division by zero
    p_norms[p_norms == 0] = 1.0
    t_norms[t_norms == 0] = 1.0
    
    # Cosine Sim matrix
    cos_sim = dot_product / (p_norms[:, np.newaxis] * t_norms)
    
    # Average of top 5 nearest neighbors
    # Sort each row of cos_sim descending
    sorted_sims = np.sort(cos_sim, axis=1)[..., ::-1]
    scores = np.mean(sorted_sims[:, :5], axis=1)

    # WikiText bonus
    target_frac = sum(1 for t in targets if _wiki_like(t)) / len(targets)
    bonus = 0.25 * target_frac
    if bonus > 0:
        flags = np.fromiter((_wiki_like(docs[i]) for i in valid), dtype=bool, count=n)
        scores = scores + bonus * flags

    # (3) Rank and keep
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

    # (4) Seeded shuffle
    rng.shuffle(kept)
    return kept
