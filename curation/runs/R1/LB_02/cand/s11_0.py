"""Curator: dedup -> junk filter -> Coherence/Density filter -> domain classifier -> word-order filter -> rank -> budget -> shuffle.

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
_FUNC = {("of", "the"), ("in", "the"), ("to", "the"), (",", "and")}
_BAD_START_PUNCT = {",", ".", ";", ":", ")", "]", "%"}

CHUNK = 2000
SCORE_CHARS = 4000
BODY_TOKENS = 600
FALLBACK_THR = 1.0

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

def _order_feats(doc, tgt_bi, tgt_tri):
    toks = _body_tokens(doc)
    n = len(toks)
    if n < 20:
        return 0.0, 0.0, 0.0, 0.0, 0
    bis = list(zip(toks, toks[1:]))
    func = 0
    hit = 0
    starts = 0
    bad = 0
    for p in bis:
        if p in tgt_bi:
            hit += 1
            if p in _FUNC:
                func += 1
        elif p in _FUNC:
            func += 1
        if p[0] == ".":
            starts += 1
            b = p[1]
            if b in _BAD_START_PUNCT or b[0].islower():
                bad += 1
    tri_hit = 0
    for t in zip(toks, toks[1:], toks[2:]):
        if t in tgt_tri:
            tri_hit += 1
    return (
        100.0 * func / n,
        hit / max(1, len(bis)),
        tri_hit / max(1, n - 2),
        (bad / starts) if starts else 0.0,
        starts,
    )

def _valley_threshold(rates):
    r = np.asarray(rates, dtype=np.float64)
    if r.size < 50:
        return FALLBACK_THR
    nb = 32
    hist, edges = np.histogram(np.clip(r, 0, 7.999), bins=nb, range=(0.0, 8.0))
    h = np.convolve(hist.astype(np.float64), [0.25, 0.5, 0.25], mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0
    low_mask = centers < 1.25
    high_mask = centers >= 2.0
    if not low_mask.any() or not high_mask.any():
        return FALLBACK_THR
    lo = int(np.argmax(np.where(low_mask, h, -1)))
    hi = int(np.argmax(np.where(high_mask, h, -1)))
    min_peak = max(5.0, 0.01 * r.size)
    if h[lo] < min_peak or h[hi] < min_peak or hi - lo < 2:
        return FALLBACK_THR
    seg = h[lo + 1:hi]
    vmin = seg.min()
    if vmin > 0.5 * min(h[lo], h[hi]):
        return FALLBACK_THR
    pos = np.flatnonzero(seg <= vmin + 1e-9)
    v = lo + 1 + int(pos[len(pos) // 2])
    return float(min(2.5, max(0.3, centers[v])))

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

def _calculate_ttr(text):
    toks = _TOK.findall(text)
    if not toks: return 0.0
    unique = len(set(toks))
    return unique / len(toks)

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

    # 2. Junk Filter
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

    # --- Coherence and Density Filter ---
    # Calculate target TTR distribution
    tgt_trs = []
    for t in target_sample:
        # Take first 500 tokens
        toks = _TOK.findall(t[:6000])[:500]
        if toks:
            tgt_trs.append(len(set(toks)) / len(toks))
    
    if tgt_trs:
        tgt_mean = np.mean(tgt_trs)
        tgt_std = np.std(tgt_trs) if len(tgt_trs) > 1 else 0.1
        
        # Filter candidate based on TTR
        valid_cand_indices = []
        for k, i in enumerate(cand):
            toks = _TOK.findall(docs[i][:6000])[:500]
            if not toks: continue
            ttr = len(set(toks)) / len(toks)
            
            # Gaussian weight: higher score if TTR is closer to target distribution
            # We prioritize docs with 0.4 < TTR < 0.8 usually, but let's use the Gaussian
            score = np.exp(-((ttr - tgt_mean)**2) / (2 * (tgt_std**2 + 0.01)))
            
            # Penalize extreme values: very low TTR (repetition) or very high TTR (gibberish)
            if ttr < 0.15 or ttr > 0.9:
                score *= 0.1
            
            # Only keep if score isn't completely decimated
            if score > 0.01:
                valid_cand_indices.append((k, i, score))
        
        # Sort by Gaussian score descending
        valid_cand_indices.sort(key=lambda x: x[2], reverse=True)
        # Take top 80% of candidates to keep variety but prune garbage
        cand = [x[1] for x in valid_cand_indices[:int(len(cand)*0.8)]]
    else:
        # Fallback if target_sample is empty/junk
        cand = cand

    # 3. Domain score
    p = _domain_scores(docs, cand, target_sample, dedup_idx, rng)

    # 4. Word-order score
    tgt_bi = set()
    tgt_tri = set()
    for t in target_sample:
        toks = _TOK.findall(t)
        tgt_bi.update(zip(toks, toks[1:]))
        tgt_tri.update(zip(toks, toks[1:], toks[2:]))

    feats = {}
    header_rates = []
    for k, i in enumerate(cand):
        d = docs[i]
        hdr = _has_header(d)
        if p[k] > 0.5 or hdr:
            f = _order_feats(d, tgt_bi, tgt_tri)
            feats[i] = f
            if hdr:
                header_rates.append(f[0])
    thr = _valley_threshold(header_rates)

    # 5. Filter
    passing = [(k, i) for k, i in enumerate(cand) if p[k] > 0.5 and feats[i][0] >= thr]
    if passing:
        med_tri = float(np.median([feats[i][2] for _, i in passing]))
        kept = []
        for k, i in passing:
            f = feats[i]
            if f[4] >= 4 and f[3] > 0.5 and f[2] < 0.25 * med_tri:
                continue
            kept.append((k, i))
        if kept:
            passing = kept

    if not passing:
        passing = [(k, i) for k, i in enumerate(cand) if i in feats and feats[i][0] >= thr]
        if not passing:
            passing = list(enumerate(cand))

    passing.sort(key=lambda ki: -p[ki[0]])
    ranked = [i for _, i in passing]

    final = _take_budget(docs, ranked, int(1.01 * T))

    # 6. Seeded shuffle
    random.Random(seed).shuffle(final)
    return final
