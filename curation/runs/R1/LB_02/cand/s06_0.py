"""Curator: dedup -> junk filter -> domain classifier -> Local Coherence -> budget -> shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled (we never pad with junk).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_SPAM = ("discount code", "coupon", "buy now", "free shipping")
_LETTERS = bytes(range(ord("a"), ord("z") + 1)) + bytes(range(ord("A"), ord("Z") + 1)) + b" "
_TOK = re.compile(r"\w+|[^\w\s]")

CHUNK = 2000
SCORE_CHARS = 4000
BODY_TOKENS = 600

def _has_header(d):
    return d.lstrip().startswith("= ")

def _chunks(text, max_chunks=None):
    out = []
    for s in range(0, len(text), CHUNK):
        c = text[s:s + CHUNK]
        if len(c) < 200 and out:
            break
        out.append(c)
        if max_chunks is not None and len(out) >= max_chunks:
            break
    return out

def _body_tokens(doc):
    s = doc.strip()
    nl = s.find("\n")
    body = s[nl + 1:] if (nl >= 0 and len(s) - nl > 100) else s
    return _TOK.findall(body[:6000])[:BODY_TOKENS]

def _domain_scores(docs, cand, target_sample, dedup_idx, rng):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        pos = []
        for t in target_sample:
            pos.extend(_chunks(t))
        neg_pool = [i for i in dedup_idx if not _has_header(docs[i])]
        if len(neg_pool) < 20:
            neg_pool = list(dedup_idx)
        neg_ids = rng.sample(neg_pool, min(3000, len(neg_pool)))
        neg = []
        for i in neg_ids:
            neg.extend(_chunks(docs[i], max_chunks=2))
        if not pos or not neg:
            return np.ones(len(cand))
        vec = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2,
            max_features=200_000, dtype=np.float64,
        )
        Xtr = vec.fit_transform(pos + neg)
        y = np.array([1] * len(pos) + [0] * len(neg))
        clf = LogisticRegression(C=10.0, max_iter=300, class_weight="balanced")
        clf.fit(Xtr, y)
        p = np.empty(len(cand), dtype=np.float64)
        B = 5000
        for s in range(0, len(cand), B):
            X = vec.transform([docs[i][:SCORE_CHARS] for i in cand[s:s + B]])
            p[s:s + B] = clf.predict_proba(X)[:, 1]
        return p
    except Exception:
        return np.ones(len(cand))

def _take_budget(docs, ranked, budget):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None
    out = []
    total = 0
    B = 1000
    for s in range(0, len(ranked), B):
        batch = ranked[s:s + B]
        if enc is not None:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in batch])]
            except Exception:
                lens = [max(1, len(docs[i]) // 4) for i in batch]
        else:
            lens = [max(1, len(docs[i]) // 4) for i in batch]
        for i, n in zip(batch, lens):
            out.append(i)
            total += n
            if total >= budget:
                return out
    return out

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedup
    seen = set()
    dedup_idx = []
    for i, d in enumerate(docs):
        key = (" ".join(d[:1000].split()).lower()[:300], len(d) // 500)
        if key in seen:
            continue
        seen.add(key)
        dedup_idx.append(i)

    # 2. Junk Filters
    cand = []
    for i in dedup_idx:
        d = docs[i]
        if len(d) < 200:
            continue
        b = d[:20000].encode("utf-8", "ignore")
        if not b:
            continue
        if len(b.translate(None, _LETTERS)) / len(b) > 0.4:
            continue
        low = d.lower()
        if any(m in low for m in _SPAM):
            continue
        cand.append(i)

    if not cand:
        idx = list(dedup_idx) if dedup_idx else list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # 3. Domain Scores
    p = _domain_scores(docs, cand, target_sample, dedup_idx, rng)

    # 4. Local Coherence Score
    # Compute target bigram distribution
    tgt_bi_counts = {}
    tgt_bi_total = 0
    for t in target_sample:
        toks = _TOK.findall(t)
        for i in range(len(toks) - 1):
            bi = (toks[i], toks[i+1])
            tgt_bi_counts[bi] = tgt_bi_counts.get(bi, 0) + 1
            tgt_bi_total += 1
    
    # Average bigram distribution (frequency)
    if tgt_bi_total > 0:
        avg_dist = {bi: count / tgt_bi_total for bi, count in tgt_bi_counts.items()}
    else:
        avg_dist = {}

    # Calculate median overlap for target_sample
    target_overlaps = []
    for t in target_sample:
        toks = _TOK.findall(t)
        count = 0
        total = 0
        for i in range(len(toks) - 1):
            bi = (toks[i], toks[i+1])
            total += 1
            if bi in avg_dist:
                count += 1
        if total > 0:
            target_overlaps.append(count / total)
    
    med_overlap = float(np.median(target_overlaps)) if target_overlaps else 0.0
    threshold = 0.4 * med_overlap

    # Filter candidates
    passing = []
    for k, i in enumerate(cand):
        if p[k] > 0.5:
            toks = _body_tokens(docs[i])
            count = 0
            total = 0
            for j in range(len(toks) - 1):
                bi = (toks[j], toks[j+1])
                total += 1
                if bi in avg_dist:
                    count += 1
            
            overlap = (count / total) if total > 0 else 0.0
            if overlap >= threshold:
                passing.append((k, i))

    if not passing:
        passing = list(enumerate(cand))

    # Rank by domain probability
    passing.sort(key=lambda ki: -p[ki[0]])
    ranked = [i for _, i in passing]

    # 5. Budgeting
    final = _take_budget(docs, ranked, int(1.01 * T))

    # 6. Final Shuffle
    random.Random(seed).shuffle(final)
    return final
