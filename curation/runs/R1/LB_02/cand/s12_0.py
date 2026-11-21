"""Curator: dedup -> junk filter -> local neighborhood similarity -> word-order filter -> budget -> shuffle.

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
SCORE_CHARS = 1000
BODY_TOKENS = 600
FALLBACK_THR = 1.0

def _has_header(d):
    return d.lstrip().startswith("= ")

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

def _local_neighborhood_score(docs, cand_indices, target_sample):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Prepare target vectors
        target_texts = [t[:SCORE_CHARS] for t in target_sample]
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
        target_matrix = vectorizer.fit_transform(target_texts)

        # Prepare candidate vectors
        cand_texts = [docs[i][:SCORE_CHARS] for i in cand_indices]
        cand_matrix = vectorizer.transform(cand_texts)

        # Compute similarities
        sims = cosine_similarity(cand_matrix, target_matrix)
        
        # For each candidate, find average of 10 nearest neighbors
        scores = []
        for i in range(len(cand_indices)):
            row = sims[i]
            # Get indices of top 10 similarities
            # Using partition is more efficient than sort
            if len(row) > 10:
                idx = np.argpartition(row, -10)[-10:]
                scores.append(np.mean(row[idx]))
            else:
                scores.append(np.mean(row))
        return np.array(scores)
    except Exception:
        return np.ones(len(cand_indices))

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

    # 1. Dedup (first occurrence wins).
    seen = set()
    dedup_idx = []
    for i, d in enumerate(docs):
        key = (" ".join(d[:1000].split()).lower()[:300], len(d) // 500)
        if key in seen:
            continue
        seen.add(key)
        dedup_idx.append(i)

    # 2. Hard junk filters.
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

    # 3. Local Neighborhood Similarity.
    sim_scores = _local_neighborhood_score(docs, cand, target_sample)

    # 4. Word-order score.
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
        # Use a slightly lowered bar for domain since we rely on neighborhood similarity
        if sim_scores[k] > 0.1 or hdr:
            f = _order_feats(d, tgt_bi, tgt_tri)
            feats[i] = f
            if hdr:
                header_rates.append(f[0])
    
    thr = _valley_threshold(header_rates)

    # 5. Keep in-domain docs that pass the order filter.
    passing = []
    for k, i in enumerate(cand):
        if i in feats:
            f = feats[i]
            # Check if it passes the order threshold
            if f[0] >= thr:
                passing.append((k, i))

    if not passing:
        # Fallback
        passing = list(enumerate(cand))

    # Sort by neighborhood similarity
    passing.sort(key=lambda ki: -sim_scores[ki[0]])
    ranked = [i for _, i in passing]

    final = _take_budget(docs, ranked, int(1.01 * T))

    # 6. Seeded shuffle.
    random.Random(seed).shuffle(final)
    return final
