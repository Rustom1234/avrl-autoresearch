"""Curator: target-similarity filter with two-stage selection logic.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. char n-gram TF-IDF + logistic regression (target_sample vs. random pool subset)
  4. Two-stage selection:
     a. Stage 1: Include all docs with score > dynamic_threshold (targeting ~80% of T)
     b. Stage 2: Fill remaining 20% of T with highest-scoring docs that 
        were excluded by the 'short-doc' rule (score <= 0.1 and length < 300).
     c. Final selection is shuffled.
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
MAX_FEATURES = 200_000
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

    # Token Estimation
    tok_est = {}
    if time.time() - t0 < HARD_DEADLINE - 25:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            B = 1000
            exact_acc = 0
            for s in range(0, len(keep), B):
                if time.time() - t0 > HARD_DEADLINE or exact_acc >= T:
                    break
                batch = [docs[i][:trunc] for i in keep[s:s+B]]
                outs = enc.encode_ordinary_batch(batch)
                for j, o in zip(range(len(batch)), outs):
                    c = len(o) + 1
                    tok_est[keep[s + j]] = c
                    exact_acc += c
        except: pass

    # Selection logic
    order = np.argsort(-scores, kind="stable")
    need = FILL * T
    
    # Find dynamic threshold for ~80% of T
    # We use lens/4 as a proxy for tokens to find the threshold quickly
    proxy_tokens = np.array([tok_est.get(keep[i], lens[i]/4.0) for i in range(len(keep))])
    sorted_proxies = sorted(proxy_tokens, reverse=True)
    target_80 = 0.8 * need
    
    # Find smallest score index that hits 80%
    thresh = 0.0
    cum_sum = 0.0
    for i, p in enumerate(sorted_proxies):
        cum_sum += p
        if cum_sum >= target_80:
            thresh = scores[order[i]] # This is a simplified heuristic
            break
    # Actually, it's better to find the threshold such that 
    # sum(tokens if score > thresh) >= 0.8 * need
    # Let's just find the score at the 80th percentile of the scores
    # but that doesn't account for token counts correctly.
    # Let's use a fixed threshold and fall back to the order.
    # To be safe and follow instructions: 
    # find a threshold that keeps ~80% of tokens
    # Using a loop to find the exact score threshold:
    possible_scores = sorted(list(set(scores)), reverse=True)
    best_thresh = 0.0
    for s_val in possible_scores:
        current_t = sum(proxy_tokens[i] for i in range(len(keep)) if scores[i] > s_val)
        if current_t >= target_80:
            best_thresh = s_val
            break
    
    # Stage 1: scores > best_thresh
    final_selection = []
    current_acc = 0.0
    stage1_indices = []
    for i in order:
        doc_idx = keep[i]
        score = scores[i]
        if score > best_thresh:
            stage1_indices.append(doc_idx)
            current_acc += tok_est.get(doc_idx, lens[i]/4.0)
            if current_acc >= need:
                final_selection = stage1_indices
                break
    else:
        final_selection = stage1_indices

    # Stage 2: Fill remaining with highest scores that were excluded by "short-doc" rule
    # Exclusion rule: score <= 0.1 and length < MIN_CHARS
    if current_acc < need:
        stage2_candidates = []
        for i in order:
            doc_idx = keep[i]
            score = scores[i]
            is_short = len(docs[doc_idx]) < MIN_CHARS
            is_low_score = score <= 0.1
            
            # If it was "bad" (short & low) and not already in selection, it's a candidate
            # Actually, instructions say "excluded by the short-doc rule". 
            # That means the "bad" ones are the ones we can't use? 
            # "fill... with highest-scoring docs that were excluded by the 'short-doc' rule"
            # This implies the "bad" ones (short and low score) are the source.
            if is_short and is_low_score:
                if doc_idx not in final_selection:
                    stage2_candidates.append(doc_idx)
        
        for doc_idx in stage2_candidates:
            final_selection.append(doc_idx)
            current_acc += tok_est.get(doc_idx, lens[keep[order.index(i)]]/4.0) # approximation
            if current_acc >= need:
                break

    # If still under, just take the best available
    if current_acc < need:
        for i in order:
            doc_idx = keep[i]
            if doc_idx not in final_selection:
                final_selection.append(doc_idx)
                current_acc += tok_est.get(doc_idx, lens[i]/4.0)
                if current_acc >= need:
                    break

    # Truncate to ensure we don't exceed need significantly (optional but good)
    # The list is already ordered by score, so we are good.
    
    rng.shuffle(final_selection)
    return final_selection
