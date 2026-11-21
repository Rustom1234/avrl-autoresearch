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

    # 1 & 2: Duplicates and Junk
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
        lines = [l.strip() for l in text.split("\n")]
        lines = [l for l in lines if l]
        if lines:
            top_count = Counter(lines).most_common(1)[0][1]
            if top_count >= 3 and top_count / len(lines) > 0.3:
                continue
        words = norm[:12000].split()[:1000]
        if len(words) >= 50 and len(set(words)) / len(words) < 0.2:
            continue
        clean.append(i)

    # 3: Word-shuffled / Damaged
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

    # 4: Domain Similarity (Logistic Regression)
    bonus_raw = np.array([_marker_bonus(docs[i]) for i in survivors], dtype=np.float64)
    base = np.full(len(survivors), np.nan, dtype=np.float64)
    
    try:
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression

        hv = HashingVectorizer(analyzer="char", ngram_range=(1, 5), n_features=2**18,
                               norm=None, alternate_sign=False, lowercase=False,
                               dtype=np.float32)

        neg_idx = random.sample([i for i in range(n) if isinstance(docs[i], str)], 
                                 min(3000, n))
        pos_texts = [t[:3000] for t in target_sample if isinstance(t, str) and t.strip()]

        if pos_texts and neg_idx:
            neg_texts = [docs[i][:3000] for i in neg_idx]
            X_counts = hv.transform(pos_texts + neg_texts)
            tfidf = TfidfTransformer(sublinear_tf=True)
            X = tfidf.fit_transform(X_counts).tocsr()
            y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))

            clf = LogisticRegression(C=10.0, solver="liblinear", dual=True,
                                     class_weight="balanced", max_iter=100)
            clf.fit(X, y)

            chunk = 500
            for s in range(0, len(survivors), chunk):
                if elapsed() > 140: break
                part = survivors[s : s + chunk]
                Xc = tfidf.transform(hv.transform([docs[i][:3000] for i in part]))
                base[s : s + len(part)] = clf.decision_function(Xc)
    except:
        pass

    base = np.nan_to_num(base, nan=0.0)
    total_scores = (base * 0.8) + (bonus_raw * 2.0)

    # 5: Priority Sorting
    has_marker = np.array([('=' in docs[i]) or ('@-@' in docs[i]) for i in survivors])
    m_indices = [survivors[i] for i, m in enumerate(has_marker) if m]
    nm_indices = [survivors[i] for i, m in enumerate(has_marker) if not m]
    
    m_scores = total_scores[has_marker]
    nm_scores = total_scores[~has_marker]
    
    m_sorted = [m_indices[i] for i in np.argsort(-m_scores, kind='stable')]
    nm_sorted = [nm_indices[i] for i in np.argsort(-nm_scores, kind='stable')]
    
    ranked = m_sorted + nm_sorted

    # 6: Selection
    selected_indices = []
    current_tokens = 0
    
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        for idx in ranked:
            tokens = len(enc.encode_ordinary(docs[idx]))
            if current_tokens + tokens <= T:
                selected_indices.append(idx)
                current_tokens += tokens
            if current_tokens > T: break
    except:
        # Fallback to rough estimation
        for idx in ranked:
            approx = len(docs[idx].split())
            if current_tokens + approx <= T:
                selected_indices.append(idx)
                current_tokens += approx
            if current_tokens > T: break

    if not selected_indices:
        selected_indices = ranked[:max(1, T // 100)] # Safety fallback

    # Final step: ensure exactly enough or cycle
    if len(selected_indices) == 0: return [ranked[0]]
    
    # Return to satisfy the logic of ordered list of indices into docs
    return selected_indices
