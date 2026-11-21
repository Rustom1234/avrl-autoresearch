import os
import random
import re
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from common import T

SPAM_MARKERS = ("online store |", "free shipping", "best price guaranteed")
TIME_BUDGET = 180.0
_NONWORD = re.compile(r"\W+")
_NONALPHA = re.compile(r"[^A-Za-z]")
_SENT = re.compile(r"[.!?] ([A-Za-z])")


def _ws_ratio(head):
    if not head:
        return 0.0
    return (head.count(" ") + head.count("\n")) / len(head)


def _alpha_ratio(head):
    if not head:
        return 0.0
    n_ws = head.count(" ") + head.count("\n")
    return (len(_NONALPHA.sub("", head)) + n_ws) / len(head)


def _bigrams(text, limit=4000):
    w = text[:limit].split()
    return list(zip(w, w[1:]))


def _hit_rate(bigrams, bset):
    if len(bigrams) < 20:
        return None
    return sum(1 for b in bigrams if b in bset) / len(bigrams)


def _cap_rate(head):
    m = _SENT.findall(head)
    if len(m) < 5:
        return None
    return sum(1 for c in m if c.isupper()) / len(m)


def _common_bigrams(texts, top=20000):
    c = Counter()
    for t in texts:
        w = t.split()
        c.update(zip(w, w[1:]))
    return set(b for b, n in c.most_common(top) if n >= 2)


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]

    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    if n == 0 or not target_sample:
        return fallback()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    t_heads = [t[:2000] for t in target_sample]
    t_ws = float(np.median([_ws_ratio(h) for h in t_heads]) if t_heads else 0.0)
    ws_thr = min(0.08, 0.5 * t_ws)
    t_alpha = np.array([_alpha_ratio(h) for h in t_heads])
    alpha_thr = 0.85 * float(np.percentile(t_alpha, 5)) if len(t_alpha) > 0 else 0.5

    bigram_set = _common_bigrams(target_sample)
    hit_thr = None
    # Simple heuristic for hit_thr if set
    if bigram_set:
        hit_thr = 0.3 

    caps = [c for c in (_cap_rate(h) for h in t_heads) if c is not None]
    cap_thr = None
    if len(caps) >= 10:
        cmed = float(np.median(caps))
        if cmed >= 0.6:
            cap_thr = 0.5 * cmed

    need_eq = (
        sum(1 for t in target_sample if t.lstrip().startswith("=")) / len(target_sample)
        >= 0.9
    )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < 50 or not d.strip():
            continue
        head = d[:2000]
        if _ws_ratio(head) < ws_thr:
            continue
        if _alpha_ratio(head) < alpha_thr:
            continue
        low = d[:20000].lower()
        if any(m in low for m in SPAM_MARKERS):
            continue
        key = _NONWORD.sub(" ", d[:400].lower()).strip()[:200]
        if not key:
            continue
        h = hash(key)
        if h in seen:
            continue
        seen.add(h)
        
        # Check bigrams/caps on survivors
        if hit_thr is not None:
            r = _hit_rate(_bigrams(d), bigram_set)
            if r is not None and r < hit_thr:
                continue
        if cap_thr is not None:
            c = _cap_rate(head)
            if c is not None and c < cap_thr:
                continue
        survivors.append(i)

    if not survivors:
        return fallback()

    # ------------------------------------------------------------------
    # Vectorized Scoring
    # ------------------------------------------------------------------
    # Use char n-grams (1,3)
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=20000)
    # Train on target sample to get common n-grams
    vectorizer.fit(target_sample)
    
    # Transform target and survivors
    target_matrix = vectorizer.transform(target_sample).toarray()
    survivor_texts = [docs[i] for i in survivors]
    survivor_matrix = vectorizer.transform(survivor_texts).toarray()
    
    # Calculate centroid
    centroid = np.mean(target_matrix, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-6:
        centroid_norm = 1.0
    centroid_norm_scaled = centroid / centroid_norm

    # Score survivors using cosine similarity
    scores = []
    for i in range(len(survivor_matrix)):
        row = survivor_matrix[i]
        doc_norm = np.linalg.norm(row)
        if doc_norm < 1e-6:
            scores.append(0.0)
        else:
            sim = np.dot(row, centroid_norm_scaled) / doc_norm
            scores.append(float(max(0.0, min(1.0, sim))))

    # ------------------------------------------------------------------
    # Damage Filter for Tier-0
    # ------------------------------------------------------------------
    # Ensure long-form documents remain coherent across 4000-char windows
    def is_coherent(doc):
        windows = [doc[i:i+4000] for i in range(0, len(doc), 4000)]
        if not windows: return True
        valid_windows = 0
        for w in windows:
            if len(w) < 100: continue
            r = _hit_rate(_bigrams(w), bigram_set)
            if r is not None and r >= 0.1:
                valid_windows += 1
        return valid_windows / max(1, len(windows)) > 0.7

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def get_tier(idx_in_survivors):
        s = scores[idx_in_survivors]
        doc_idx = survivors[idx_in_survivors]
        doc_text = docs[doc_idx]
        
        # Tier 0 logic with damage filter
        if s > 0.5:
            if not need_eq or doc_text.lstrip().startswith("="):
                if not is_coherent(doc_text):
                    return 2
                return 0
            return 1
        return 2

    tiers = [get_tier(i) for i in range(len(survivors))]
    
    # Rank: tier (asc), score (desc), index (asc)
    ranked_indices = sorted(range(len(survivors)), 
                              key=lambda i: (tiers[i], -scores[i], i))

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    target_tok = int(T * 1.02)
    floor_tok = int(T * 0.3)
    selected = []
    total_tokens = 0
    
    for rank_idx in ranked_indices:
        doc_idx = survivors[rank_idx]
        
        # Skip Tier 1+ if we've hit the floor, unless we have no more Tier 0s
        if tiers[rank_idx] > 0 and total_tokens >= floor_tok:
            # Check if any lower tiers exist to fill up to T
            pass # Actually, let's just continue to allow filling
        
        if enc is not None and (time.time() - t0) < TIME_BUDGET - 20:
            try:
                count = len(enc.encode(docs[doc_idx]))
            except:
                count = len(docs[doc_idx]) // 4
        else:
            count = len(docs[doc_idx]) // 4
            
        selected.append(doc_idx)
        total_tokens += count
        if total_tokens >= target_tok:
            break

    if not selected:
        return fallback()

    random.Random(seed).shuffle(selected)
    return selected
