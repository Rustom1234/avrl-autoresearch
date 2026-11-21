"""Curator: Pointwise ranking via TF-IDF centroid cosine similarity.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. TF-IDF Cosine Similarity to Target Centroid:
     a. Vectorize target_sample and pool docs
     b. Calculate mean vector (centroid) of target_sample
     c. Calculate cosine similarity of all pool docs to centroid
  4. Selection:
     a. Filter out "junk" (short + low similarity)
     b. Sort by similarity descending
     c. Fill until budget T * 1.02 is met
"""
import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

MAX_CHARS = 3000
MIN_CHARS = 300
MIN_UNIQ_RATIO = 0.3
MAX_BAD_FRAC = 0.3
FILL = 1.02
MAX_FEATURES = 50_000

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
    
    # 1. Dedupe
    seen = set()
    uniq_idx = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h not in seen:
            seen.add(h)
            uniq_idx.append(i)

    # 2. Strict Junk Guards
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(uniq_idx)
        rng.shuffle(out)
        return out

    use_len = sum(len(t) < MIN_CHARS for t in targets) / len(targets) <= 0.10
    use_uniq = sum(_uniq_ratio(t) < MIN_UNIQ_RATIO for t in targets) / len(targets) <= 0.10
    use_bad = sum(_bad_frac(t) > MAX_BAD_FRAC for t in targets) / len(targets) <= 0.10

    keep = []
    for i in uniq_idx:
        d = docs[i]
        if use_len and len(d) < MIN_CHARS: continue
        if use_bad and _bad_frac(d) > MAX_BAD_FRAC: continue
        if use_uniq and _uniq_ratio(d) < MIN_UNIQ_RATIO: continue
        keep.append(i)
    
    if not keep: return []

    # 3. Pointwise Ranking via Centroid Cosine Similarity
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    # To optimize memory/speed, truncate documents to MAX_CHARS
    target_texts = [t[:MAX_CHARS] for t in targets]
    pool_texts = [docs[i][:MAX_CHARS] for i in keep]
    
    X_target = vectorizer.fit_transform(target_texts)
    X_pool = vectorizer.transform(pool_texts)
    
    # Calculate Centroid
    centroid = np.asarray(X_target.mean(axis=0))
    
    # Cosine Similarity: (X_pool * centroid) / (norm(X_pool) * norm(centroid))
    # Since X_pool is already L2 normalized by TfidfVectorizer (if we used norm='l2')
    # but sublinear_tf changes things, we calculate manually:
    
    # Dot product
    dot = np.dot(X_pool, centroid)
    
    # Norms
    # Sum of squares of non-zero elements for L2 norm
    def get_l2_norm(matrix):
        return np.sqrt(np.sum(matrix**2, axis=1))
    
    norm_pool = get_l2_norm(X_pool)
    norm_centroid = np.linalg.norm(centroid)
    
    # Avoid division by zero
    similarities = np.divide(dot, norm_pool * norm_centroid, 
                               out=np.zeros_like(dot), 
                               where=(norm_pool * norm_centroid) != 0)

    # 4. Selection Logic
    # Sort by similarity descending
    sorted_indices = np.argsort(-similarities, kind="stable")
    
    final_selection = []
    current_tokens = 0
    target_tokens = T * FILL
    
    # Fast token count approximation (chars / 4) to avoid tiktoken overhead
    # if we want to stay strictly within time/memory limits.
    for i in sorted_indices:
        doc_idx = keep[i]
        sim = similarities[i]
        
        # Skip docs that are very short AND have very low similarity
        if len(docs[doc_idx]) < MIN_CHARS and sim < 0.1:
            continue
            
        final_selection.append(doc_idx)
        # Approximate token count (standard for small models: ~4 chars per token)
        current_tokens += len(docs[doc_idx]) / 4.0
        
        if current_tokens >= target_tokens:
            break
            
    rng.shuffle(final_selection)
    return final_selection
