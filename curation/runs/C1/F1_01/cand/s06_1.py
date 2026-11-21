```python
"""Curator: filter + rank pipeline.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. drop exact / near duplicates (hash of normalised full text and of its first 200 chars)
  2. drop junk (too short, repeated boilerplate lines, low unique-word ratio)
  3. drop word-shuffled / damaged docs (bigram overlap with the target sample,
     plus a "capitalised token after ' . '" ordering proxy)
  4. score domain similarity (char 1-5-gram hashed TF-IDF + logistic regression,
     target sample vs. random pool docs) and add a bonus for wikitext markers
  5. take the top docs that meet the in-domain threshold (ranked_scores >= thr).
     If < T, fill remaining capacity with docs from the remaining pool,
     skipping every 5th highest-ranked doc to ensure diversity.
  6. seeded shuffle of the selection
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
import hashlib
import warnings
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:
    T = 9_830_400

TIME_SCORE_END = 140.0
TIME_SCORE_HARD = 150.0
TIME_TOKEN_HARD = 166.0
FIT_RESERVE = 25.0

def _marker_bonus(text):
    b = 0.0
    if text.startswith('= '):
        b += 0.5
    if ' @-@ ' in text:
        b += 0.15
    if ' @,@ ' in text:
        b += 0.15
    if ' , ' in text:
        b += 0.15
    return b

def _body_after_title(text):
    if text.startswith('= '):
        nl = text.find('\n')
        if nl >= 0:
            return text[nl + 1:]
    return text

def select(docs, target_sample):
    start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    n = len(docs)
    if n == 0:
        return []

    def elapsed():
        return time.time() - start

    seen_full = set()
    seen_head = set()
    deduped = []
    clean = []
    
    for i, text in enumerate(docs):
        if not isinstance(text, str):
            continue
        norm = " ".join(text.lower().split())
        if not norm:
            continue
        k_full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        k_head = hashlib.md5(norm[:200].encode("utf-8", "ignore")).digest()
        if k_full in seen_full or k_head in seen_head:
            continue
        seen_full.add(k_full)
        seen_head.add(k_head)
        deduped.append(i)

        if len(text.strip()) < 200:
            continue
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            top = Counter(lines).most_common(1)[0][1]
            if top >= 3 and top / len(lines) > 0.3:
                continue
        words = norm[:12000].split()[:1000]
        if len(words) >= 50 and len(set(words)) / len(words) < 0.2:
            continue
        clean.append(i)

    target_bigrams = set()
    for t in target_sample:
        toks = t.lower().split()
        target_bigrams.update(zip(toks, toks[1:]))

    fracs = {}
    cap_ok = {}
    for i in clean:
        text = docs[i]
        toks = _body_after_title(text)[:30000].split()[:2000]
        low = [w.lower() for w in toks[:400]]
        nb = len(low) - 1
        if nb >= 1:
            hit = 0
            for bg in zip(low, low[1:]):
                if bg in target_bigrams:
                    hit += 1
            fracs[i] = hit / nb
        else:
            fracs[i] = 0.0
        nd = 0
        nc = 0
        last = len(toks) - 1
        for j, w in enumerate(toks):
            if w == '.' and j < last:
                nd += 1
                if toks[j + 1][:1].isupper():
                    nc += 1
        cap_ok[i] = True if nd < 5 else (nc / nd > 0.5)

    survivors = clean
    if clean and target_bigrams:
        ref = [fracs[i] for i in clean if docs[i].startswith('= ')]
        if len(ref) < 20:
            ref = [fracs[i] for i in clean]
        thr_bg = 0.5 * float(np.median(ref)) if ref else 0.0
        survivors = [i for i in clean if fracs[i] >= thr_bg and cap_ok[i]]
    if not survivors:
        survivors = clean if clean else (deduped if deduped else list(range(n)))

    bonus_raw = np.array([_marker_bonus(docs[i]) for i in survivors], dtype=np.float64)
    base = np.full(len(survivors), np.nan, dtype=np.float64)
    thr = -np.inf
    scale = 1.0
    
    try:
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        hv = HashingVectorizer(analyzer="char", ngram_range=(1, 5), n_features=2**18,
                               norm=None, alternate_sign=False, lowercase=False, dtype=np.float32)
        neg_idx = [i for i in random.sample(range(n), min(3000, n)) if isinstance(docs[i], str)]
        pos_texts_full = [t for t in target_sample if isinstance(t, str) and t.strip()]
        
        if pos_texts_full and neg_idx:
            probe = [t[:3000] for t in pos_texts_full]
            t0 = time.time()
            hv.transform(probe)
            rate = 1.25 * (time.time() - t0) / max(1, sum(len(p) for p in probe))
            lens = np.array([len(docs[i]) for i in neg_idx] + [len(docs[i]) for i in survivors] + 
                      [len(t) for t in pos_texts_full], dtype=np.int64)
            avail = max(8.0, TIME_SCORE_END - elapsed() - FIT_RESERVE)
            cap = 300
            for c in (3000, 2500, 2000, 1500, 1200, 1000, 800, 600, 450, 300):
                if rate * float(np.minimum(lens, c)).sum() <= avail:
                    cap = c
                    break
            pos_texts = [t[:cap] for t in pos_texts_full]
            neg_texts = [docs[i][:cap] for i in neg_idx]
            X_counts = hv.transform(pos_texts + neg_texts)
            tfidf = TfidfTransformer(sublinear_tf=True)
            X = tfidf.fit_transform(X_counts).tocsr()
            y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
            clf = LogisticRegression(C=10.0, solver="liblinear", dual=True, class_weight="balanced", 
                                    max_iter=200, random_state=seed).fit(X, y)
            
            oof = None
            if len(pos_texts_full) >= 6 and len(neg_texts) >= 6 and elapsed() < TIME_SCORE_END - 40:
                try:
                    k = 3
                    fold = np.zeros(len(pos_texts_full) + len(neg_texts), dtype=np.int64)
                    pp = list(range(len(pos_texts_full)))
                    nn = list(range(len(pos_texts_full), len(pos_texts_full) + len(neg_texts)))
                    r_shuffle = random.Random(seed + 1)
                    r_shuffle.shuffle(pp)
                    r_shuffle.shuffle(nn)
                    for j, r in enumerate(pp): fold[r] = j % k
                    for j, r in enumerate(nn): fold[r] = j % k
                    oof = np.zeros(len(pos_texts_full) + len(neg_texts), dtype=np.float64)
                    for f in range(k):
                        tr = np.where(fold != f)[0]
                        te = np.where(fold == f)[0]
                        m = LogisticRegression(C=10.0, solver="liblinear", dual=True, class_weight="balanced", 
                                              max_iter=100, random_state=seed).fit(X[tr], y[tr])
                        oof[te] = m.decision_function(X[te])
                except: oof = None
            
            chunk = 500
            for s in range(0, len(survivors), chunk):
                if elapsed() > TIME_SCORE_HARD: break
                part = survivors[s:s+chunk]
                Xc = tfidf.transform(hv.transform([docs[i][:cap] for i in part]))
                base[s:s+len(part)] = clf.decision_function(Xc)
            
            if oof is not None:
                neg_oof = {i: oof[len(pos_texts_full) + j] for j, i in enumerate(neg_idx)}
                for p, i in enumerate(survivors):
                    if i in neg_oof and not np.isnan(base[p]): base[p] = neg_oof[i]
                target_oof = oof[:len(pos_texts_full)]
            
            done = ~np.isnan(base)
            if done.any():
                sd = float(np.std(base[done]))
                scale = sd if sd > 1e-9 else 1.0
                if not done.all(): base[~done] = float(np.min(base[done])) - scale
            else: base[:] = 0.0
            
            if target_oof is not None:
                t_bonus = np.array([_marker_bonus(t) for t in pos_texts_full])
                t_total = target_oof + scale * t_bonus
                thr = float(np.percentile(t_total, 5))
        else:
            base[:] = 0.0
            thr = -np.inf
    except:
        base = np.where(np.isnan(base), 0.0, base)
        thr = -np.inf
        scale = 1.0

    base = np.where(np.isnan(base), 0.0, base)
    total = base + scale * bonus_raw
    order = np.argsort(-total, kind="stable")
    ranked = [survivors[p] for p in order]
    ranked_scores = total[order]

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except: pass

    all_counts = []
    if enc is not None and elapsed() < TIME_TOKEN_HARD:
        try:
            batch_size = 512
            for s in range(0, len(ranked), batch_size):
                part = ranked[s:s+batch_size]
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in part])]
                all_counts.extend(counts)
        except: pass
    
    if not all_counts:
        all_counts = [max(1, len(docs[i]) // 4) for i in ranked]

    selected = []
    tot = 0
    in_domain_indices = [i for i, s in enumerate(ranked_scores) if s >= thr]
    
    for i in in_domain_indices:
        if tot + all_counts[i] <= T:
            selected.append(ranked[i])
            tot += all_counts[i]
        else:
            break
            
    used_in_rank = [i for i in in_domain_indices]
    # Re-run diversity fill
    diverse_pool = [ranked[i] for i in range(len(ranked)) if i not in in_domain_indices]
    # The prompt says "take top docs... if < T, fill remaining capacity with docs from remaining pool, skipping every 5th"
    # Since 'ranked' is already sorted, we just need to skip indices in 'ranked' that aren't 'selected'
    # and weren't 'in_domain'. Actually simpler:
    
    # Let's just stick to the prompt's logic: 
    # 1. Get all that meet threshold. 2. Fill to T from remaining ranked, skipping every 5th.
    
    # Re-implementing the selection exactly as the logic step 5 says:
    # "take the top docs that meet the in-domain threshold... If < T, fill remaining capacity... skipping every 5th"
    
    # Let's fix the logic:
    final_selected = []
    final_tot = 0
    
    # Phase 1: Top docs meeting threshold
    # Since ranked is sorted by score, the first X docs meeting threshold are the "top" ones.
    for i in range(len(ranked)):
        if ranked_scores[i] >= thr:
            if final_tot + all_counts[i] <= T:
                final_selected.append(ranked[i])
                final_tot += all_counts[i]
            else:
                break
    
    # Phase 2: Fill from remaining
    # "Remaining pool" usually means docs not yet selected. 
    # The most logical order is to continue through the ranked list.
    # We skip every 5th highest-ranked doc from the pool (ranked list).
    
    # Let's track which ranked indices are already in final_selected
    selected_rank_indices = set()
    # This is inefficient if T is huge, but ranked is at most a few thousand usually.
    # Since docs are unique-ish, we can use ranks.
    # Wait, let's just track by rank index.
    # Because we want the "top" docs meeting threshold, we need to know their rank indices.
    
    # Let's restart the selection logic to be 100% safe to the prompt.
    # 1. Find all rank indices where ranked_scores[i] >= thr
    # 2. Add them to final_selected as long as they fit.
    # 3. From the remaining rank indices, add them skipping every 5th.
    
    # Re-do:
    final_selected = []
    final_tot = 0
    rank_indices = list(range(len(ranked)))
    
    # This is a bit tricky because we need to maintain the "top" order.
    # Let's just do it in two passes.
    
    # Pass 1:
    taken_ranks = set()
    for i in range(len(rank_indices)):
        if ranked_scores[i] >= thr:
            if final_tot + all_counts[i] <= T:
                final_selected.append(ranked[i])
                final_tot += all_counts[i]
                taken_ranks.add(i)
            else:
                break
    
    # Pass 2:
    # Remaining pool is all rank indices not in taken_ranks
    remaining_ranks = [i for i in rank_indices if i not in taken_ranks]
    # "skipping every 5th" - this usually refers to the order in the ranked list.
    #
