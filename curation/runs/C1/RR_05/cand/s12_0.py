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

    # Calibration
    t_heads = [t[:2000] for t in target_sample]
    t_ws = float(np.median([_ws_ratio(h) for h in t_heads]))
    ws_thr = min(0.08, 0.5 * t_ws)
    t_alpha = np.array([_alpha_ratio(h) for h in t_heads])
    alpha_thr = 0.85 * float(np.percentile(t_alpha, 5))

    bigram_set = _common_bigrams(target_sample)
    hit_thr = None
    fold_a = target_sample[0::2]
    fold_b = target_sample[1::2]
    real_rates, shuf_rates = [], []
    if fold_a and fold_b:
        for train, test in ((fold_a, fold_b), (fold_b, fold_a)):
            bs = _common_bigrams(train)
            for t in test:
                r = _hit_rate(_bigrams(t), bs)
                if r is not None: real_rates.append(r)
                w = t[:4000].split()
                rng.shuffle(w)
                r = _hit_rate(list(zip(w, w[1:])), bs)
                if r is not None: shuf_rates.append(r)
    if len(real_rates) >= 10:
        med = float(np.median(real_rates))
        hit_thr = 0.5 * med
        if len(shuf_rates) >= 10:
            s95 = float(np.percentile(shuf_rates, 95))
            r05 = float(np.percentile(real_rates, 5))
            if s95 < r05:
                hit_thr = 0.5 * (s95 + r05)

    caps = [c for c in (_cap_rate(h) for h in t_heads) if c is not None]
    cap_thr = None
    if len(caps) >= 10:
        cmed = float(np.median(caps))
        if cmed >= 0.6:
            cap_thr = 0.5 * cmed

    need_eq = (sum(1 for t in target_sample if t.lstrip().startswith("=")) / len(target_sample) >= 0.9)

    # Filtering
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < 50 or not d.strip(): continue
        head = d[:2000]
        if _ws_ratio(head) < ws_thr or _alpha_ratio(head) < alpha_thr: continue
        low = d[:20000].lower()
        if any(m in low for m in SPAM_MARKERS): continue
        key = _NONWORD.sub(" ", d[:400].lower()).strip()[:200]
        if not key: continue
        h = hash(key)
        if h in seen: continue
        seen.add(h)
        if hit_thr is not None:
            r = _hit_rate(_bigrams(d), bigram_set)
            if r is not None and r < hit_thr: continue
        if cap_thr is not None:
            c = _cap_rate(head)
            if c is not None and c < cap_thr: continue
        survivors.append(i)

    if not survivors: return fallback()

    # Vectorized Scoring
    vocab_counter = Counter()
    for t in target_sample:
        text = t[:5000].lower()
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                vocab_counter[text[i:i+n]] += 1
    common_ngrams = [ngram for ngram, count in vocab_counter.most_common(20000)]
    
    texts_target = [t[:5000].lower() for t in target_sample]
    texts_survivors = [docs[i][:5000].lower() for i in survivors]
    
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), 
                                  vocabulary=common_ngrams, lowercase=True)
    X_target = vectorizer.fit_transform(texts_target).toarray()
    X_survivors = vectorizer.transform(texts_survivors).toarray()
    
    centroid = np.mean(X_target, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    centroid_norm_scaled = centroid / centroid_norm if centroid_norm > 1e-6 else centroid
    
    survivor_norms = np.linalg.norm(X_survivors, axis=1)
    similarities = np.dot(X_survivors, centroid_norm_scaled) / (survivor_norms + 1e-9)
    scores = {survivors[i]: float(max(0.0, min(1.0, similarities[i]))) for i in range(len(survivors))}

    # Cross-chunk consistency
    target_hit_rates = [r for r in _hit_rate(_bigrams(t), bigram_set) if r is not None for t in target_sample]
    global_hit_rate = np.mean(target_hit_rates) if target_hit_rates else 0
    hit_std = np.std(target_hit_rates) if len(target_hit_rates) > 1 else 0

    def check_consistency(i):
        d = docs[i]
        chunks = [d[j:j+4000] for j in range(0, len(d), 4000)]
        rates = []
        for c in chunks:
            r = _hit_rate(_bigrams(c), bigram_set)
            rates.append(r if r is not None else global_hit_rate)
        if not rates: return True
        if any(r < (global_hit_rate - 3 * hit_std) for r in rates): return False
        first_third_avg = np.mean(rates[:max(1, len(rates)//3)])
        last_third_avg = np.mean(rates[-max(1, len(rates)//3):])
        if first_third_avg > 0 and last_third_avg < first_third_avg * 0.5: return False
        return True

    def tier(i):
        s = scores[i]
        if s > 0.5:
            if not need_eq or docs[i].lstrip().startswith("="): return 0
            return 1
        return 2

    tiers = {i: tier(i) for i in survivors}
    valid_survivors = []
    for i in survivors:
        if tiers[i] == 0 and not check_consistency(i): continue
        valid_survivors.append(i)
    
    order = sorted(valid_survivors, key=lambda i: (tiers[i], -scores[i], i))

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    target_tok = int(T * 1.02)
    floor_tok = int(T * 0.3)
    selected = []
    total = 0
    done = False
    batch_size = 128
    
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        counts = None
        if enc is not None and (time.time() - t0) < TIME_BUDGET - 30:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in chunk])]
            except: counts = None
        
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in chunk]
            
        for i, c in zip(chunk, counts):
            if tiers[i] > 0 and total >= floor_tok:
                done = True
                break
            selected.append(i)
            total += c
            if total >= target_tok:
                done = True
                break
        if done: break

    if not selected: return fallback()
    random.Random(seed).shuffle(selected)
    return selected
