"""Curator: centroid-based similarity filter using TF-IDF and Cosine Similarity.

Pipeline:
  1. exact-dedupe by normalized text hash.
  2. strict junk guards (length, unique-word ratio, non-alpha fraction).
  3. TF-IDF Vectorization (char_wb) of target_sample and pool docs.
  4. Scoring: Cosine similarity to the centroid of the target_sample.
  5. Selection: Sorted by similarity, applying a "short doc" penalty.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from common import T

MAX_CHARS = 3000          # chars of each doc seen by the vectorizer / guards
MAX_FEATURES = 50000      # Reduced for memory efficiency
MIN_CHARS = 300           # Minimum length for useful content
MIN_UNIQ_RATIO = 0.3      # Minimum unique word ratio
MAX_BAD_FRAC = 0.3        # Maximum non-alpha fraction
FILL = 1.02               # fraction of T to select
HARD_DEADLINE = 160.0     # stop optional work

_GOOD_RE = re.compile(r"[A-Za-z\s]+")

def _norm_hash(text):
    norm = " ".join(text.lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()

def _bad_frac(text):
    s = text[:MAX_CHARS]
    if not s: return 1.0
    return len(_GOOD_RE.sub("", s)) / len(s)

def _uniq_ratio(text):
    words = text[:MAX_CHARS].lower().split()
    if not words: return 0.0
    return len(set(words)) / len(words)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1. Dedupe
    seen = set()
    uniq = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h in seen: continue
        seen.add(h)
        uniq.append(i)

    # 2. Strict Junk Guards
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    nt = max(1, len(targets))
    
    use_len = sum(len(t) < MIN_CHARS for t in targets) / nt <= 0.10
    use_uniq = sum(_uniq_ratio(t) < MIN_UNIQ_RATIO for t in targets) / nt <= 0.10
    use_bad = sum(_bad_frac(t) > MAX_BAD_FRAC for t in targets) / nt <= 0.10

    keep = []
    for i in uniq:
        d = docs[i]
        if use_len and len(d) < MIN_CHARS: continue
        if use_bad and _bad_frac(d) > MAX_BAD_FRAC: continue
        if use_uniq and _uniq_ratio(d) < MIN_UNIQ_RATIO: continue
        keep.append(i)
    if not keep: keep = uniq

    if not targets:
        out = list(keep)
        rng.shuffle(out)
        return out

    # 3. Centroid Similarity Scoring
    # Build TF-IDF on targets to find the "ideal" document profile
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    # Fit on targets (the "positive" class)
    # Note: To have a better centroid, we could fit on pool+targets, 
    # but fitting on targets is more memory efficient and captures the "target" profile.
    X_tgt = vec.fit_transform(targets)
    # Calculate centroid: mean of rows
    # For sparse matrix, mean across rows:
    tgt_sum = X_tgt.sum(axis=0)
    centroid = tgt_sum / len(targets)
    # Ensure centroid is a sparse row vector for dot product
    centroid = sparse.csr_matrix(centroid.real)

    # Score pool docs
    # Process in chunks to avoid memory spikes if the pool is huge
    scores = np.zeros(len(keep), dtype=np.float64)
    chunk_size = 2000
    for s in range(0, len(keep), chunk_size):
        chunk_indices = keep[s : s + chunk_size]
        chunk_texts = [docs[i][:MAX_CHARS] for i in chunk_indices]
        X_chunk = vec.transform(chunk_texts)
        # Cosine Similarity (normalized dot product)
        # Since TF-IDF is already L2 normalized (or nearly so with sublinear_tf),
        # a simple dot product is a very good proxy for cosine similarity.
        # We want to normalize the centroid to unit length first.
        centroid_norm = np.linalg.norm(centroid.data)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
            
        chunk_scores = X_chunk.dot(centroid).toarray().flatten()
        scores[s : s + len(chunk_indices)] = chunk_scores

    # 4. Selection Logic
    order = np.argsort(-scores, kind="stable")
    need = FILL * T
    
    final_selection = []
    current_acc = 0.0
    
    # Heuristic for token count: chars / 4
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        
        # Penalty: skip short docs unless they have some relevance
        if len(docs[doc_idx]) < MIN_CHARS and score < 0.1:
            continue
            
        final_selection.append(doc_idx)
        current_acc += len(docs[doc_idx]) / 4.0
        if current_acc >= need:
            break

    # Token Refinement
    tok_counts = {}
    if time.time() - t0 < HARD_DEADLINE - 30:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            # Only do exact counts if we have time and for a limited number of items
            # to ensure we don't time out.
            limit = min(len(final_selection), 1000)
            for idx in final_selection[:limit]:
                # This is a slow operation, so we only do it if we have plenty of time
                if time.time() - t0 > HARD_DEADLINE - 10: break
                tok_counts[idx] = len(enc.encode_ordinary_batch([docs[idx]])[0]) + 1
        except ImportError:
            pass

    # Re-filter with accurate token counts if available
    final_selection_accurate = []
    acc_accurate = 0.0
    for idx in final_selection:
        count = tok_counts.get(idx, len(docs[idx]) / 4.0)
        final_selection_accurate.append(idx)
        acc_accurate += count
        if acc_accurate >= need:
            break
    
    final_selection = final_selection_accurate

    rng.shuffle(final_selection)
    return final_selection
