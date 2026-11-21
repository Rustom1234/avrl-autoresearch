"""Curator: Centroid-based ranking with strict junk guards.

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. Pointwise Ranking:
     a. Calculate the centroid (mean vector) of the target_sample using TF-IDF
     b. Score every surviving pool doc with the dot product against this centroid
  4. selection:
     a. Filter out "short/low-score" junk
     b. Fill until T tokens are met
     c. Seeded shuffle of the selected indices

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import T

MAX_CHARS = 3000          # chars of each doc seen by the vectorizer / guards
N_NEG = 8000              # pool docs used for potential selection
MAX_FEATURES = 150_000
MIN_CHARS = 300           # Minimum useful length
MIN_UNIQ_RATIO = 0.3      # Minimum unique words ratio
MAX_BAD_FRAC = 0.3        # Maximum non-alpha fraction
FILL = 1.02               # fraction of T to select
VEC_BUDGET = 90.0         # seconds allotted to n-gram extraction
HARD_DEADLINE = 155.0     # stop optional work (tiktoken)

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

    # 3. Pointwise Ranking (Centroid-based)
    vec = _make_vectorizer(MAX_FEATURES)
    
    # Fit on targets to get the centroid
    X_targets = vec.fit_transform([t[:MAX_CHARS] for t in targets])
    centroid = np.mean(X_targets, axis=0)
    
    # Score the kept docs
    # Process in chunks if memory is an issue, but pool is usually manageable
    scores = np.zeros(len(keep), dtype=np.float32)
    chunk_size = 1000
    for s in range(0, len(keep), chunk_size):
        end = min(s + chunk_size, len(keep))
        chunk_indices = keep[s:end]
        X_chunk = vec.transform([docs[i][:MAX_CHARS] for i in chunk_indices])
        scores[s:end] = np.dot(X_chunk, centroid)

    # 4. Selection Logic
    # Sort by scores descending
    order = np.argsort(-scores, kind="stable")
    need = FILL * T
    
    final_selection = []
    current_acc = 0.0
    
    # Score mapping for quick lookup
    # We filter out "short/low-score" junk
    # A doc is a "bad" candidate if it's short AND has a low score (below some threshold)
    # Since we are using dot products, we'll use a relative threshold or just favor score
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        
        # Filter: skip docs that are short and don't meet a minimum quality score
        if len(docs[doc_idx]) < MIN_CHARS and score < 0.05:
            continue
            
        final_selection.append(doc_idx)
        # Rough token count estimate (chars / 4)
        current_acc += len(docs[doc_idx]) / 4.0
        if current_acc >= need:
            break

    # Optional Token Refinement
    tok_map = {}
    if time.time() - t0 < HARD_DEADLINE - 20:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            # Only refine if we haven't met the count yet to save time
            if current_acc < need:
                for idx in final_selection:
                    if idx not in tok_map:
                        tok_map[idx] = len(enc.encode_ordinary_batch([docs[idx]])[0]) + 1
                        current_acc += tok_map[idx]
                        if current_acc >= need: break
        except: pass

    # Final pass for exact count using tiktoken counts if available
    final_selection = []
    current_acc = 0.0
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        if len(docs[doc_idx]) < MIN_CHARS and score < 0.05:
            continue
            
        final_selection.append(doc_idx)
        c = tok_map.get(doc_idx, len(docs[doc_idx]) // 4 + 1)
        current_acc += c
        if current_acc >= need:
            break

    rng.shuffle(final_selection)
    return final_selection
