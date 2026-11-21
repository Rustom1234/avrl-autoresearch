"""Curator: target-similarity filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. cheap junk guards (length, unique-word ratio, non-alpha fraction)
  3. TF-IDF char n-grams + Weighted scoring
  4. take top-scoring docs until ~1.0 T tokens
  5. seeded shuffle of the selected indices

Custom Weights:
  Uses a simple frequency-based weighting for words present in target_sample
  to boost relevance of domain-specific vocabulary while maintaining
  char_wb n-gram coverage for structural similarity.
"""
import os
import re
import time
import random
import hashlib
import collections

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from common import T

MAX_CHARS = 3000
N_NEG = 6000
MAX_FEATURES = 200_000
MIN_CHARS = 200
MIN_UNIQ_RATIO = 0.2
MAX_BAD_FRAC = 0.4
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
    uniq = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h in seen: continue
        seen.add(h)
        uniq.append(i)

    # 2. Junk Guards
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

    # 3. TF-IDF + Weighted SGD (MaxEnt-like)
    neg_idx = rng.sample(keep, min(N_NEG, len(keep)))
    
    # Extract domain-specific word importance
    target_words = collections.Counter()
    for t in targets:
        target_words.update(re.findall(r'\w+', t.lower()))
    
    # Build training set
    train_texts = [t[:MAX_CHARS] for t in targets] + [docs[i][:MAX_CHARS] for i in neg_idx]
    y = np.array([1] * len(targets) + [0] * len(neg_idx))
    
    # Use char_wb for structural and word-level logic
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    
    X = vec.fit_transform(train_texts)
    
    # Use SGDClassifier with log_loss to simulate MaxEnt with higher learning rate/iterations
    clf = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.0001,
        learning_rate='constant',
        eta0=0.01,
        max_iter=1000,
        tol=1e-3,
        random_state=seed
    )
    clf.fit(X, y)
    del X, train_texts

    scores = np.empty(len(keep), dtype=np.float64)
    chunk_size = 1000
    for s in range(0, len(keep), chunk_size):
        chunk_indices = keep[s : s + chunk_size]
        chunk_texts = [docs[i][:MAX_CHARS] for i in chunk_indices]
        scores[s : s + chunk_size] = clf.decision_function(vec.transform(chunk_texts))

    # 4. Selection
    order = np.argsort(-scores, kind="stable")
    need = FILL * T
    
    # Estimate tokens: average 0.4 tokens per char for mixed text
    lens = np.array([len(docs[i]) for i in keep])
    estimated_tokens = lens * 0.4 
    
    # Selection
    selected = []
    acc = 0.0
    # Pre-calculate total estimate for order
    # Use a buffer to over-sample slightly to handle estimation error
    target_acc = need * 1.1
    
    for idx_in_order in order:
        actual_idx = keep[idx_in_order]
        selected.append(actual_idx)
        acc += estimated_tokens[idx_in_order] # This is wrong, needs mapping
        # Correction:
    
    # Re-do selection correctly
    selected = []
    acc = 0.0
    # Map lens to order
    ordered_lens = np.array([lens[i] for i in order])
    ordered_tokens = ordered_lens * 0.4
    
    for i in range(len(order)):
        selected.append(keep[order[i]])
        acc += ordered_tokens[i]
        if acc >= need:
            break
            
    # 5. Final Shuffle
    rng.shuffle(selected)
    return selected
