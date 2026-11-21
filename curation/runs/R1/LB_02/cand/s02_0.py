"""Curator: dedup -> junk filter -> tf-idf similarity -> word-order filter -> budget -> shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens.
If the list holds fewer than T tokens it is cycled (we never pad with junk).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import numpy as np
from collections import Counter

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

def _get_tfidf_sim(docs, cand_indices, target_sample):
    """Computes cosine similarity to target_sample using a simple TF-IDF on first SCORE_CHARS."""
    def get_counts(text):
        words = _TOK.findall(text[:SCORE_CHARS].lower())
        return Counter(words)

    target_counts = []
    for t in target_sample:
        target_counts.append(get_counts(t))

    # Simple TF-IDF components
    all_words = set()
    for t in target_sample:
        all_words.update(_TOK.findall(t[:SCORE_CHARS].lower()))
    for i in cand_indices:
        all_words.update(_TOK.findall(docs[i][:SCORE_CHARS].lower()))

    # Simplified IDF (log(N/df))
    df = {}
    for w in all_words:
        count = 0
        for t in target_sample:
            if w in _TOK.findall(t[:SCORE_CHARS].lower()): count += 1
        df[w] = count
    
    # Target centroids (average counts for target)
    centroid = Counter()
    for tc in target_counts:
        for w, c in tc.items():
            centroid[w] += c / len(target_counts)
            
    scores = []
    for i in cand_indices:
        doc_counts = get_counts(docs[i])
        dot = 0.0
        norm_doc = 0.0
        norm_target = 0.0
        for w, count in doc_counts.items():
            idf = np.log(len(target_sample) / (1 + df.get(w, 0)))
            dot += count * idf * centroid.get(w, 0)
            norm_doc += (count * idf) ** 2
        
        for w, count in centroid.items():
            idf = np.log(len(target_sample) / (1 + df.get(w, 0)))
            norm_target += (count * idf) ** 2
            
        scores.append(dot / (np.sqrt(norm_doc) * np.sqrt(norm_target) + 1e-9))
    
    return np.array(scores)

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

    seen = set()
    dedup_idx = []
    for i, d in enumerate(docs):
        key = (" ".join(d[:1000].split()).lower()[:300], len(d) // 500)
        if key in seen:
            continue
        seen.add(key)
        dedup_idx.append(i)

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

    p = _get_tfidf_sim(docs, cand, target_sample)

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

    passing = [(k, i) for k, i in enumerate(cand) if p[k] > 0.5 and i in feats and feats[i][0] >= thr]
    
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

    rng.shuffle(final)
    return final
