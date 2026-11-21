"""Curator: target-similarity filter with strict junk guards.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. char n-gram TF-IDF + logistic regression (target_sample vs. random pool subset),
     score every surviving pool doc with decision_function
  4. selection:
     a. high-scoring docs (score > 0.1) are taken first
     b. short docs (< 300 chars) are filtered unless high-scoring
     c. fill remaining tokens with next best scores
  5. seeded shuffle of the selected indices

The seed for any randomness is the environment variable CURATE_SEED.
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

MAX_CHARS = 3000          # chars of each doc seen by the vectorizer / guards
N_NEG = 6000              # random pool docs used as negatives
MAX_FEATURES = 200_000
MIN_CHARS = 300           # Raised from 200
MIN_UNIQ_RATIO = 0.3      # Tightened from 0.2
MAX_BAD_FRAC = 0.3        # Tightened from 0.4
FILL = 1.02               # fraction of T to select
VEC_BUDGET = 95.0         # seconds allotted to n-gram extraction
HARD_DEADLINE = 160.0     # stop optional work (tiktoken)

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
    order = np.argsort(-scores, kind="stable")
    need = FILL * T
    
    # First pass: high quality scores and long docs
    # Second pass: anything else until budget is met
    selected_indices = []
    acc = 0.0
    
    # Use tokens/4 as a heuristic for pre-selection
    # Then refine with exact count if time permits
    potential_candidates = []
    for idx_in_keep in order:
        doc_idx = keep[idx_in_keep]
        score = scores[idx_in_keep]
        # Criteria: High score (>0.1) OR long enough doc
        if score > 0.1 or len(docs[doc_idx]) >= MIN_CHARS:
            potential_candidates.append(doc_idx)
            
    # Re-sort potential candidates by score to ensure best first
    # We map scores back to the doc_idx
    score_map = {keep[i]: scores[i] for i in range(len(keep))}
    potential_candidates.sort(key=lambda x: score_map[x], reverse=True)

    # Refine candidates to meet need
    # Note: order list is already sorted by score
    # We can just iterate 'order' but skip if it's a "short doc" with a "low score"
    final_selection = []
    current_acc = 0.0
    
    # Simplified selection to respect the constraints efficiently:
    # 1. Take all docs with score > 0.1
    # 2. Take docs with length >= MIN_CHARS
    # 3. If still under need, take anything else
    
    # Let's just do a single pass over 'order' and apply the "short doc" rule:
    # A doc is skipped if it's < MIN_CHARS AND score <= 0.1
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        is_short = len(docs[doc_idx]) < MIN_CHARS
        is_low_score = score <= 0.1
        
        if is_short and is_low_score:
            continue
            
        final_selection.append(doc_idx)
        # Heuristic weight
        current_acc += lens[i] / 4.0
        if current_acc >= need:
            break

    # Token refinement
    tok = {}
    if time.time() - t0 < HARD_DEADLINE - 25:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            B = 1000
            exact_acc = 0
            for s in range(0, len(final_selection), B):
                if time.time() - t0 > HARD_DEADLINE or exact_acc >= need:
                    break
                batch = final_selection[s:s + B]
                outs = enc.encode_ordinary_batch([docs[j] for j in batch])
                for j, o in zip(batch, outs):
                    c = len(o) + 1
                    tok[j] = c
                    exact_acc += c
        except: pass

    final_selection = []
    acc = 0.0
    for idx in final_selection: # wait, logic error, need to iterate the filtered list
        pass # logic corrected below
    
    # Corrected selection loop
    final_selection = []
    acc = 0.0
    # Re-iterate the candidates that passed the junk filter
    # The "order" list already contains the best scores
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        if len(docs[doc_idx]) < MIN_CHARS and score <= 0.1:
            continue
        
        final_selection.append(doc_idx)
        acc += tok.get(doc_idx, lens[i] / 4.0)
        if acc >= need:
            break

    rng.shuffle(final_selection)
    return final_selection
