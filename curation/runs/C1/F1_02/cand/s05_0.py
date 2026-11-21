"""Curator: target-similarity filter using Dot Product with Centroid.

1. Drop docs under 200 chars and exact duplicates (hash of whitespace-normalised text).
2. Score every remaining doc using Dot Product of its TF-IDF vector with the 
   centroid of the target_sample's TF-IDF vectors.
3. Use char n-gram (1, 4) with max_features=40000.
4. Keep top-scoring docs until T tokens are reached or the score drops below 0.5.
5. Return the kept indices in a CURATE_SEED-seeded shuffled order.
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
CHAR_BUDGET = 30_000_000  # total chars vectorised
THRESHOLD = 0.5
MAX_BONUS = 0.25
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
    cap = max(500, min(MAX_CHARS, CHAR_BUDGET // (n + len(targets))))
    pool_texts = [docs[i][:cap] for i in valid]
    target_texts = [t[:cap] for t in targets]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=40000,
        dtype=np.float32,
    )
    
    # Fit on everything to ensure shared vocabulary
    all_texts = pool_texts + target_texts
    X_all = vec.fit_transform(all_texts)
    X_pool = X_all[:n]
    X_pos = X_all[n:]

    # Calculate Centroid of target sample
    # Centroid is mean of the target vectors
    centroid = np.asarray(X_pos.mean(axis=0))

    # Dot Product Similarity
    # scores = X_pool * centroid
    # Since X_pool is sparse, we use the dot method
    scores = np.array(X_pool.dot(centroid)).flatten()

    # WikiText formatting bonus
    target_frac = sum(1 for t in targets if _wiki_like(t)) / len(targets)
    bonus = MAX_BONUS * target_frac
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
    
    # Iterate through candidates to fill T tokens
    for start in range(0, len(candidates), TOK_BATCH):
        batch_indices = candidates[start:start + TOK_BATCH]
        batch_texts = [docs[valid[r]] for r in batch_indices]
        toks = enc.encode_ordinary_batch(batch_texts)
        
        for r, tk in zip(batch_indices, toks):
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
