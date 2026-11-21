import os
import random
import re
import time
from collections import Counter

import numpy as np
from common import T

SPAM_MARKERS = ("online store |", "free shipping", "best price guaranteed")
TIME_BUDGET = 180.0
_NONWORD = re.compile(r"\W+")
_NONALPHA = re.compile(r"[^A-Za-z]")
_SENT = re.compile(r"[.!?] ([A-Za-z])")

def _ws_ratio(head):
    if not head: return 0.0
    return (head.count(" ") + head.count("\n")) / len(head)

def _alpha_ratio(head):
    if not head: return 0.0
    n_ws = head.count(" ") + head.count("\n")
    return (len(_NONALPHA.sub("", head)) + n_ws) / len(head)

def _bigrams(text, limit=4000):
    w = text[:limit].split()
    return list(zip(w, w[1:]))

def _hit_rate(bigrams, bset):
    if len(bigrams) < 20: return None
    return sum(1 for b in bigrams if b in bset) / len(bigrams)

def _cap_rate(head):
    m = _SENT.findall(head)
    if len(m) < 5: return None
    return sum(1 for c in m if c.isupper()) / len(m)

def _common_bigrams(texts, top=20000):
    c = Counter()
    for t in texts:
        w = t.split()
        c.update(zip(w, w[1:]))
    return set(b for b, n in c.most_common(top) if n >= 2)

def _style_score(text):
    limit = text[:5000]
    words = limit.split()
    if not words: return 0.0
    target_chars = sum(limit.count(c) for c in [' ', ',', '.', '@'])
    return target_chars / (len(limit) / 1000.0)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]

    if n == 0 or not target_sample:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # 1. Calibration & Prep
    t_heads = [t[:2000] for t in target_sample]
    t_ws = float(np.median([_ws_ratio(h) for h in t_heads]))
    ws_thr = min(0.08, 0.5 * t_ws)
    t_alpha = np.array([_alpha_ratio(h) for h in t_heads])
    alpha_thr = 0.85 * float(np.percentile(t_alpha, 5))

    bigram_set = _common_bigrams(target_sample)
    hit_thr = None
    real_rates, shuf_rates = [], []
    fold_a, fold_b = target_sample[0::2], target_sample[1::2]
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
            hit_thr = 0.5 * (float(np.percentile(shuf_rates, 95)) + float(np.percentile(real_rates, 5)))

    caps = [c for c in (_cap_rate(h) for h in t_heads) if c is not None]
    cap_thr = None
    if len(caps) >= 10 and float(np.median(caps)) >= 0.6:
        cap_thr = 0.5 * float(np.median(caps))

    need_eq = (sum(1 for t in target_sample if t.lstrip().startswith("=")) / len(target_sample)) >= 0.9

    # 2. Filter & Dedupe
    raw_survivors = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < 50 or not d.strip(): continue
        head = d[:2000]
        if _ws_ratio(head) < ws_thr or _alpha_ratio(head) < alpha_thr: continue
        if any(m in d[:20000].lower() for m in SPAM_MARKERS): continue
        
        r = _hit_rate(_bigrams(d), bigram_set)
        if hit_thr is not None and r is not None and r < hit_thr: continue
        c = _cap_rate(head)
        if cap_thr is not None and c is not None and c < cap_thr: continue
        
        raw_survivors.append((i, d))

    # Dedupe: Keep longest doc per key
    groups = {}
    for i, d in raw_survivors:
        key = _NONWORD.sub(" ", d[:400].lower()).strip()[:200]
        h = hash(key)
        if h not in groups or len(d) > len(docs[groups[h]]):
            groups[h] = i
    
    deduped_indices = sorted(list(set(groups.values())))
    recovered_count = len(raw_survivors) - len(deduped_indices)
    print(f"Recovered by dedupe: {recovered_count}")

    # 3. Scoring
    common_ngrams = [ngram for ngram, count in Counter().update(
        [ngram for t in target_sample for n in range(1, 4) for i in range(len(t[:5000].lower()) - n + 1) 
         for ngram in [t[:5000].lower()[i:i+n]]]
    ).most_common(20000)] # Note: Counter doesn't work like this, using manual approach
    
    # Correct way to build common_ngrams for the scoring part:
    c_score = Counter()
    for t in target_sample:
        text = t[:5000].lower()
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                c_score[text[i:i+n]] += 1
    common_ngrams = [ng for ng, _ in c_score.most_common(20000)]
    
    target_vecs = []
    for t in target_sample:
        text = t[:5000].lower()
        vec = np.array([text.count(ng) for ng in common_ngrams], dtype=np.float32)
        target_vecs.append(vec)
    
    target_vecs = np.array(target_vecs)
    scores = {}
    for i in deduped_indices:
        text = docs[i][:5000].lower()
        vec = np.array([text.count(ng) for ng in common_ngrams], dtype=np.float32)
        norm = np.linalg.norm(vec)
        scores[i] = float(np.dot(vec, np.mean(target_vecs, axis=0)) / norm) if norm > 1e-6 else 0.0

    # Leave-one-out threshold
    loo_scores = []
    for i in range(len(target_vecs)):
        others = np.delete(target_vecs, i, axis=0)
        vec = target_vecs[i]
        norm = np.linalg.norm(vec)
        loo_scores.append(float(np.dot(vec, np.mean(others, axis=0)) / norm) if norm > 1e-6 else 0.0)
    thr = float(np.percentile(loo_scores, 5)) - 0.02
    print(f"Threshold: {thr}")

    # 4. Tiers
    target_style_5th = np.percentile([_style_score(t) for t in target_sample], 5)
    
    def get_tier(idx):
        s = scores[idx]
        if s >= thr:
            if need_eq and docs[idx].lstrip().startswith("="): return 0
            if not need_eq or not docs[idx].lstrip().startswith("="):
                if _style_score(docs[idx]) >= 0.5 * target_style_5th: return 1
            return 1
        return 2

    tiers = {i: get_tier(i) for i in deduped_indices}
    
    # Selection Logic
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        enc = None

    def get_len(idx):
        if enc:
            try: return len(enc.encode_ordinary([docs[idx]]))[0]
            except: pass
        return max(1, len(docs[idx]) // 4)

    t0_list = sorted([i for i in deduped_indices if tiers[i] == 0], key=lambda x: (-scores[x], x))
    t1_list = sorted([i for i in deduped_indices if tiers[i] == 1], key=lambda x: (-scores[x], x))
    
    selected = []
    total_toks = 0
    
    for i in t0_list:
        if total_toks + get_len(i) <= T * 1.02:
            selected.append(i)
            total_toks += get_len(i)
            
    for i in t1_list:
        if total_toks + get_len(i) <= T * 1.02:
            selected.append(i)
            total_toks += get_len(i)

    # Fallback if tiers 0 & 1 are too small
    if total_toks < T * 0.3:
        t2_list = sorted([i for i in deduped_indices if tiers[i] == 2], key=lambda x: (-scores[x], x))
        for i in t2_list:
            selected.append(i)
            total_toks += get_len(i)
            if total_toks >= T: break

    print(f"Tokens in Tier 0/1: {total_toks}")
    
    # Final Truncate and Shuffle
    final_selected = []
    curr_sum = 0
    for i in selected:
        l = get_len(i)
        if curr_sum + l <= T * 1.02:
            final_selected.append(i)
            curr_sum += l
        else: break
    
    rng.shuffle(final_selected)
    return final_selected
