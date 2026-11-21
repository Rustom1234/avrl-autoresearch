"""Curator: nearest neighbor scoring with fixed n-gram vocabulary and windowed damage filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Drop junk: empty docs, low-whitespace blobs, spam, duplicates.
  2. Drop word-shuffled / case-damaged docs using bigram hit rate and cap rate.
  3. Score survivors using cosine similarity against the centroid of the target_sample
     using a shared n-gram vocabulary.
  4. Windowed Damage Filter: Calibrate a char-LM and ngram metrics on target_sample
     to filter out noisy/repetitive docs from the top tier.
  5. Take top docs until ~T gpt2 tokens.
  6. Seeded shuffle.
"""
import os
import random
import re
import time
from collections import Counter
import math

import numpy as np
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
        rng.shuffle(idx)
        return idx

    if n == 0 or not target_sample:
        return fallback()

    # Calibration
    t_heads = [t[:2000] for t in target_sample]
    t_ws = float(np.median([_ws_ratio(h) for h in t_heads]))
    ws_thr = min(0.08, 0.5 * t_ws)
    t_alpha = np.array([_alpha_ratio(h) for h in t_heads])
    alpha_thr = 0.85 * float(np.percentile(t_alpha, 5))

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
    
    bigram_set = _common_bigrams(target_sample)
    hit_thr = None
    if len(real_rates) >= 10:
        med = float(np.median(real_rates))
        hit_thr = 0.5 * med
        if len(shuf_rates) >= 10:
            s95 = float(np.percentile(shuf_rates, 95))
            r05 = float(np.percentile(real_rates, 5))
            if s95 < r05: hit_thr = 0.5 * (s95 + r05)

    caps = [c for c in (_cap_rate(h) for h in t_heads) if c is not None]
    cap_thr = None
    if len(caps) >= 10:
        cmed = float(np.median(caps))
        if cmed >= 0.6: cap_thr = 0.5 * cmed

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

    # Similarity
    vocab_counter = Counter()
    for t in target_sample:
        text = t[:5000].lower()
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                vocab_counter[text[i:i+n]] += 1
    common_ngrams = [ngram for ngram, count in vocab_counter.most_common(20000)]
    vocab = set(common_ngrams)
    
    target_counts = np.zeros(len(common_ngrams), dtype=np.float32)
    for t in target_sample:
        text = t[:5000].lower()
        counts = np.zeros(len(common_ngrams), dtype=np.float32)
        for j, ngram in enumerate(common_ngrams):
            counts[j] = text.count(ngram)
        target_counts += counts
    
    centroid = target_counts / len(target_sample)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-6: centroid_norm = 1.0
    centroid_norm_scaled = centroid / centroid_norm

    scores = {}
    for i in survivors:
        text = docs[i][:5000].lower()
        counts = np.zeros(len(common_ngrams), dtype=np.float32)
        for j, ngram in enumerate(common_ngrams):
            counts[j] = text.count(ngram)
        doc_norm = np.linalg.norm(counts)
        scores[i] = float(max(0.0, min(1.0, np.dot(counts, centroid_norm_scaled) / doc_norm))) if doc_norm > 1e-6 else 0.0

    # Windowed Damage Filter
    if (time.time() - t0) < (TIME_BUDGET - 120):
        fold_data = [[] for _ in range(2)]
        for idx, t in enumerate(target_sample):
            fold_data[idx % 2].append(t.lower())
        
        def get_lm_counts(fold_idx):
            c = Counter()
            for t in fold_data[fold_idx]:
                for n in range(1, 4):
                    for i in range(len(t) - n + 1):
                        c[t[i:i+n]] += 1
            return c
        
        lm_counts = [get_lm_counts(0), get_lm_counts(1)]
        window_ce, window_hits, window_oov, doc_ces = [], [], [], []
        
        word_vocab = set()
        for t in target_sample:
            words = set(t.lower().split())
            word_vocab.update(words)
            
        for f_idx in range(2):
            other_idx = 1 - f_idx
            other_lm = lm_counts[other_idx]
            other_bigrams = _common_bigrams(fold_data[other_idx])
            
            for t in fold_data[f_idx]:
                doc_ce_sum = 0
                doc_windows = []
                for start in range(0, len(t), 1500):
                    win = t[start:start+1500]
                    ce_win, count_win = 0, 0
                    for n in range(1, 4):
                        for i in range(len(win) - n + 1):
                            ngram = win[i:i+n]
                            c = other_lm[ngram]
                            ce_win += math.log(1 + 10 / (c + 10))
                            count_win += 1
                    win_ce = ce_win / max(1, count_win)
                    w_parts = win.split()
                    w_bigrams = list(zip(w_parts, w_parts[1:]))
                    win_hit = _hit_rate(w_bigrams, other_bigrams) if len(w_bigrams) > 20 else 0
                    oov_count, w_count = 0, 0
                    for w in w_parts:
                        w_count += 1
                        if w not in word_vocab: oov_count += 1
                    win_oov = oov_count / max(1, w_count)
                    doc_windows.append((win_ce, win_hit, win_oov))
                    window_ce.append(win_ce)
                    window_hits.append(win_hit if win_hit is not None else 0)
                    window_oov.append(win_oov)
                doc_ces.append(sum(w[0] for w in doc_windows) / len(doc_windows))

        ce_thr = np.percentile(window_ce, 99) + 0.05
        hit_thr_w = np.percentile(window_hits, 1) * 0.9
        oov_thr = np.percentile(window_oov, 1) * 0.9
        doc_ce_thr = np.percentile(doc_ces, 99)

        def is_damaged(idx):
            d = docs[idx]
            seen_chunks = set()
            for i in range(0, len(d) - 200, 200):
                chunk = d[i:i+200]
                if chunk in seen_chunks: return True
                seen_chunks.add(chunk)
            t_low = d.lower()
            doc_windows_data = []
            doc_ce_sum = 0
            step = max(1, len(t_low) // 40)
            for i in range(40):
                start = i * step
                win = t_low[start:start+1500]
                if len(win) < 10: continue
                ce_win, count_win = 0, 0
                for n in range(1, 4):
                    for j in range(len(win) - n + 1):
                        ngram = win[j:j+n]
                        c = other_lm[ngram]
                        ce_win += math.log(1 + 10 / (c + 10))
                        count_win += 1
                win_ce = ce_win / max(1, count_win)
                w_parts = win.split()
                w_bigrams = list(zip(w_parts, w_parts[1:]))
                win_hit = _hit_rate(w_bigrams, other_bigrams) if len(w_bigrams) > 20 else 0
                oov_count, w_count = 0, 0
                for w in w_parts:
                    w_count += 1
                    if w not in word_vocab: oov_count += 1
                win_oov = oov_count / max(1, w_count)
                doc_windows_data.append((win_ce, win_hit, win_oov))
                doc_ce_sum += win_ce
            if not doc_windows_data: return False
            mean_doc_ce = doc_ce_sum / len(doc_windows_data)
            if mean_doc_ce > doc_ce_thr: return True
            violation_count = 0
            for w_ce, w_hit, w_oov in doc_windows_data:
                if w_ce > ce_thr or (w_hit is not None and w_hit < hit_thr_w) or (w_oov > oov_thr):
                    violation_count += 1
            if violation_count / len(doc_windows_data) > 0.15: return True
            return False

        new_survivors = []
        for i in survivors:
            s = scores[i]
            if s > 0.5:
                if not need_eq or docs[i].lstrip().startswith("="):
                    if not is_damaged(i): new_survivors.append(i)
                else: new_survivors.append(i)
            else: new_survivors.append(i)
        survivors = new_survivors

    def tier(i):
        s = scores[i]
        if s > 0.5:
            if not need_eq or docs[i].lstrip().startswith("="): return 0
            return 1
        return 2

    tiers = {i: tier(i) for i in survivors}
    order = sorted(survivors, key=lambda i: (tiers[i], -scores[i], i))

    selected = []
    current_tokens = 0
    for i in order:
        d_tokens = len(docs[i].split()) # rough estimate
        if current_tokens + d_tokens > T and current_tokens > 0:
            continue
        selected.append(i)
        current_tokens += d_tokens
        if current_tokens >= T: break
    
    if not selected: return fallback()
    return selected
