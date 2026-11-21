"""Curator: Pointwise Ranking via Cosine Similarity to Target Centroid.

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. Pointwise Ranking:
     a. Build char n-gram TF-IDF for target_sample and a subset of the pool.
     b. Compute the centroid of target_sample vectors.
     c. Score every surviving pool doc by cosine similarity to the centroid.
  4. Selection:
     a. Score documents; prioritize high scores and long content.
     b. Fill until T tokens are reached.
     c. Final shuffle.
"""
import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MAX_CHARS = 3000          
N_NEG = 4000              # Pool size to help define the feature space
MAX_FEATURES = 100_000
MIN_CHARS = 300           
MIN_UNIQ_RATIO = 0.3      
MAX_BAD_FRAC = 0.3        
FILL = 1.02               
VEC_BUDGET = 120.0         
HARD_DEADLINE = 160.0     

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

def _make_vectorizer(max_features):
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )

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

    # 3. Pointwise Ranking via Cosine Similarity
    # Use a representative subset of the pool to fit the vectorizer
    # if the pool is massive, to ensure the vocabulary is grounded.
    subset_indices = rng.sample(keep, min(len(keep), N_NEG))
    train_corpus = [t[:MAX_CHARS] for t in targets] + [docs[i][:MAX_CHARS] for i in subset_indices]
    
    vec = _make_vectorizer(MAX_FEATURES)
    X_train = vec.fit_transform(train_corpus)
    
    # Create centroid of target sample
    # X_train[:len(targets)] are the target documents
    target_vectors = X_train[:len(targets)]
    if len(target_vectors) > 0:
        centroid = np.mean(target_vectors, axis=0).reshape(1, -1)
    else:
        centroid = np.zeros((1, MAX_FEATURES))

    # Score all kept documents
    # Process in chunks to avoid memory issues and respect time
    scores = np.zeros(len(keep))
    chunk_size = 2000
    for s in range(0, len(keep), chunk_size):
        chunk_indices = keep[s : s + chunk_size]
        chunk_texts = [docs[i][:MAX_CHARS] for i in chunk_indices]
        X_chunk = vec.transform(chunk_texts)
        # Cosine similarity to centroid
        sims = cosine_similarity(X_chunk, centroid)
        scores[s : s + len(chunk_indices)] = sims.flatten()

    # 4. Selection Logic
    order = np.argsort(-scores, kind="stable")
    need = FILL * T
    
    final_selection = []
    acc = 0.0
    
    # Heuristic: Token count approx = chars / 4
    # We want to fill 'need' tokens with high similarity docs
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        
        # Filter: Skip very short, low-quality docs unless they are somewhat relevant
        if len(docs[doc_idx]) < MIN_CHARS and score < 0.05:
            continue
            
        final_selection.append(doc_idx)
        acc += len(docs[doc_idx]) / 4.0
        if acc >= need:
            break

    # If we didn't reach the goal (e.g. many filtered out), take remaining
    if acc < need:
        remaining_pool = list(set(keep) - set(final_selection))
        rng.shuffle(remaining_pool)
        for idx in remaining_pool:
            final_selection.append(idx)
            acc += len(docs[idx]) / 4.0
            if acc >= need:
                break

    # Cap exactly at T if possible (simple truncation of indices is usually fine 
    # because the trainer stops at T tokens anyway)
    rng.shuffle(final_selection)
    return final_selection
