"""Curator: Target-similarity filter using Nearest Centroid scoring.

Pipeline:
  1. exact-dedupe by normalized text hash.
  2. strict junk guards (length, unique-word ratio, non-alpha fraction).
  3. TfidfVectorizer on target_sample and a subset of the pool.
  4. Compute mean TF-IDF vector (centroid) of target_sample.
  5. Score pool docs by cosine similarity to centroid.
  6. Select docs based on similarity, length, and token budget.
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
MIN_CHARS = 300
MIN_UNIQ_RATIO = 0.3
MAX_BAD_FRAC = 0.3
FILL = 1.02
VEC_BUDGET = 95.0
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

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1. Dedupe
    seen = set()
    uniq_indices = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h in seen: continue
        seen.add(h)
        uniq_indices.append(i)

    # 2. Junk Guards
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(uniq_indices)
        rng.shuffle(out)
        return out

    nt = len(targets)
    use_len = sum(len(t) < MIN_CHARS for t in targets) / nt <= 0.10
    use_uniq = sum(_uniq_ratio(t) < MIN_UNIQ_RATIO for t in targets) / nt <= 0.10
    use_bad = sum(_bad_frac(t) > MAX_BAD_FRAC for t in targets) / nt <= 0.10

    keep = []
    for i in uniq_indices:
        d = docs[i]
        if use_len and len(d) < MIN_CHARS: continue
        if use_bad and _bad_frac(d) > MAX_BAD_FRAC: continue
        if use_uniq and _uniq_ratio(d) < MIN_UNIQ_RATIO: continue
        keep.append(i)
    
    if not keep: keep = uniq_indices

    # 3. Tfidf & Centroid Scoring
    # Sample negatives to build the vocabulary
    neg_count = min(len(keep), 1000)
    neg_indices = rng.sample(keep, neg_count)
    
    # Determine truncation length
    trunc = 300
    for L in [3000, 2500, 2000, 1500, 1000, 500, 300]:
        test_corpus = [t[:L] for t in targets] + [docs[i][:L] for i in neg_indices]
        try:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), 
                                sublinear_tf=True, max_features=100000)
            v.fit(test_corpus)
            # Check time
            if (time.time() - t0) < VEC_BUDGET:
                trunc = L
                break
        except: pass

    train_texts = [t[:trunc] for t in targets] + [docs[i][:trunc] for i in neg_indices]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), 
                                  sublinear_tf=True, max_features=100000)
    X_train = vectorizer.fit_transform(train_texts)
    
    # Target Centroid
    X_targets = X_train[:nt]
    centroid = np.mean(X_targets, axis=0).reshape(1, -1)
    
    # Score all kept docs
    keep_texts = [docs[i][:trunc] for i in keep]
    X_keep = vectorizer.transform(keep_texts)
    scores = cosine_similarity(X_keep, centroid).flatten()

    # 4. Selection logic
    # Sort indices by score descending
    sorted_indices = np.argsort(-scores)
    
    # Pre-calculate token counts if possible
    tok_counts = {}
    if time.time() - t0 < HARD_DEADLINE - 30:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            for i, idx in enumerate(sorted_indices):
                d_idx = keep[i]
                # Only encode high confidence/large docs to save time
                if i < 2000 or scores[i] > 0.2:
                    tok_counts[d_idx] = len(enc.encode_ordinary_batch([docs[d_idx]]))[0] + 1
        except: pass

    final_selection = []
    current_acc = 0.0
    need = FILL * T
    
    # First pass: High scores and long docs
    # We iterate the sorted list (best scores first)
    for i in sorted_indices:
        idx = keep[i]
        score = scores[i]
        
        # Filter out tiny junk with low similarity
        if len(docs[idx]) < MIN_CHARS and score < 0.05:
            continue
            
        final_selection.append(idx)
        current_acc += tok_counts.get(idx, len(docs[idx]) // 4)
        if current_acc >= need:
            break

    rng.shuffle(final_selection)
    return final_selection
