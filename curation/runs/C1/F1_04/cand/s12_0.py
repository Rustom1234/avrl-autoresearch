"""Curator: weighted quality score selection with strict junk guards.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. char n-gram TF-IDF + logistic regression (target_sample vs. random pool subset)
  4. Weighted Score calculation:
     Score = (LR_score * 0.8) + ((log(len(doc)) / max_log_len) * 0.2)
  5. Selection:
     - Filter out docs that fail strict junk guards.
     - Sort by Weighted Score descending.
     - Take docs until token budget T is reached (using 4 chars/token heuristic).
     - Final shuffle.
"""
import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

MAX_CHARS = 3000
N_NEG = 6000
MAX_FEATURES = 150_000
MIN_CHARS = 300
MIN_UNIQ_RATIO = 0.3
MAX_BAD_FRAC = 0.3
FILL = 1.02
VEC_BUDGET = 90.0
HARD_DEADLINE = 150.0

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

    # 3. Logistic Regression
    neg_idx = rng.sample(keep, min(N_NEG, len(keep)))
    calib = [docs[i][:MAX_CHARS] for i in neg_idx[:60]]
    calib_chars = max(1, sum(len(c) for c in calib))
    tc = time.time()
    try:
        _make_vectorizer(None).fit_transform(calib)
    except: pass
    rate = calib_chars / max(1e-6, time.time() - tc)

    lens = np.fromiter((len(docs[i]) for i in keep), dtype=np.int64, count=len(keep))
    neg_lens = np.fromiter((len(docs[i]) for i in neg_idx), dtype=np.int64, count=len(neg_idx))
    tgt_lens = np.fromiter((len(t) for t in targets), dtype=np.int64, count=len(targets))
    
    trunc = 300
    for L in (3000, 2500, 2000, 1500, 1200, 1000, 800, 600, 450, 300):
        total = (np.minimum(lens, L).sum() + np.minimum(neg_lens, L).sum()
                 + np.minimum(tgt_lens, L).sum())
        if total / rate * 1.25 <= VEC_BUDGET:
            trunc = L
            break

    train_texts = [t[:trunc] for t in targets] + [docs[i][:trunc] for i in neg_idx]
    y = np.array([1] * len(targets) + [0] * len(neg_idx))
    vec = _make_vectorizer(MAX_FEATURES)
    X = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf.fit(X, y)
    del X, train_texts

    scores = np.empty(len(keep), dtype=np.float64)
    CH = 2000
    for s in range(0, len(keep), CH):
        chunk = [docs[i][:trunc] for i in keep[s:s + CH]]
        scores[s:s + CH] = clf.decision_function(vec.transform(chunk))

    # 4. Selection Logic
    # Calculate log length for diversity score
    log_lens = np.log1p(lens)
    max_log_len = log_lens.max() if len(log_lens) > 0 else 1.0
    
    # Weighted Score: (LR_score * 0.8) + ((log(len) / max_log_len) * 0.2)
    # Clip scores to reasonable range for mixing
    lr_scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    weighted_scores = (lr_scores_norm * 0.8) + ((log_lens / max_log_len) * 0.2)
    
    order = np.argsort(-weighted_scores, kind="stable")
    need = FILL * T
    
    final_selection = []
    acc = 0.0
    
    # Heuristic: 4 chars/token
    for i in order:
        doc_idx = keep[i]
        # Skip if doc is tiny and score is low (safety guard)
        if len(docs[doc_idx]) < 100 and weighted_scores[i] < 0.1:
            continue
            
        final_selection.append(doc_idx)
        acc += lens[i] / 4.0
        if acc >= need:
            break

    rng.shuffle(final_selection)
    return final_selection
