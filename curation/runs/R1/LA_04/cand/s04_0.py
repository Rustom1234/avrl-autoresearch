```python
"""Target-similarity curator with word-order / damage filter and cosine similarity scoring.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop obvious junk (too short, low letter ratio, low overlap with the
     target vocabulary).
  2. Word-order / damage filter: word-bigram coverage against the target
     sample, plausibility of the token following 'the', and fraction of
     whitespace tokens in the target vocabulary. Thresholds are calibrated on
     the target sample itself (half vs. half, rescaled to the full set using
     pool docs).
  3. Dedup on first-500-char hash, normalised title line and three 200-char
     windows (25/50/75%); among colliding docs keep the one with the highest
     bigram coverage, then the longest.
  4. Score the surviving docs with a TF-IDF Cosine Similarity to the target
     centroid (char_wb, n-gram (2,4), 200k features) plus a bonus for WikiText
     formatting cues.
  5. Take the best-scoring docs up to about T gpt2 tokens, but only from the
     target-like set; if that set holds at least 0.3*T tokens return only it
     (it is cycled), otherwise pad with the next best docs.
  6. Shuffle the selection with the CURATE_SEED-seeded RNG.
"""
import os
import random
import re
import time
import hashlib
import numpy as np

from common import T

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+")
_NON_ASCII_LETTER = re.compile(r"[^A-Za-z]")
_TITLE = re.compile(r"=\s+([^=\n]+?)\s+=\s*$")

PREFIX = 3000          # chars of each doc seen by the classifier / junk filter
QPREFIX = 6000         # chars of each doc seen by the word-order filter
MIN_CHARS = 200
VOCAB_FRAC_MIN = 0.5
N_CALIB_POOL = 2000
TOKEN_GOAL = 1.02      # small safety margin over T
CLEAN_MIN_FRAC = 0.3   # clean set is returned alone if it holds this much of T
TIME_LIMIT = 180.0

BAD_AFTER_THE = frozenset([",", ".", "the", "of", "and"])
MARGIN_A = 0.02
MARGIN_B = 0.03
MARGIN_C = 0.03

def _letter_ratio(s):
    nonspace = len(_WS.sub("", s))
    if nonspace == 0:
        return 0.0
    letters = len(_NON_ASCII_LETTER.sub("", s))
    return letters / nonspace

def _format_bonus(d):
    b = 0.0
    if d.lstrip()[:2] == "= ":
        b += 1.0
    if " @-@ " in d:
        b += 0.5
    if " , " in d:
        b += 0.5
    if " . " in d:
        b += 0.5
    return b

def _basic_filter(docs, target_sample):
    vocab = set()
    for t in target_sample:
        vocab.update(_WORD.findall(t.lower()))

    t_ratios = sorted(_letter_ratio(t[:PREFIX]) for t in target_sample if t.strip())
    letter_thr = 0.6
    if t_ratios:
        p5 = t_ratios[int(0.05 * (len(t_ratios) - 1))]
        letter_thr = min(letter_thr, 0.85 * p5)

    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        head = d[:PREFIX]
        if _letter_ratio(head) < letter_thr:
            continue
        words = _WORD.findall(head.lower())
        if not words:
            continue
        if vocab:
            hits = sum(1 for w in words if w in vocab)
            if hits / len(words) < VOCAB_FRAC_MIN:
                continue
        keep.append(i)
    return keep

def _toks(text):
    return text[:QPREFIX].lower().split()

def _cov(toks, bg):
    n = len(toks) - 1
    if n <= 0:
        return 0.0, 0
    return sum(map(bg.__contains__, zip(toks, toks[1:]))) / n, n

def _vfrac(toks, vocab):
    if not toks:
        return 0.0
    return sum(map(vocab.__contains__, toks)) / len(toks)

def _the_ok(toks):
    nxt = [b for a, b in zip(toks, toks[1:]) if a == "the"]
    if not nxt:
        return 1.0, 0
    bad = sum(1 for b in nxt if b in BAD_AFTER_THE)
    return 1.0 - bad / len(nxt), bad

def _build_sets(tok_lists):
    bg = set()
    vocab = set()
    for toks in tok_lists:
        vocab.update(toks)
        bg.update(zip(toks, toks[1:]))
    return bg, vocab

def _quality_filter(docs, keep, target_sample, rng):
    full_tok = [t.lower().split() for t in target_sample]
    bg_full, vocab_full = _build_sets(full_tok)

    cov = {}
    feats = {}
    for i in keep:
        toks = _toks(docs[i])
        a, n = _cov(toks, bg_full)
        b, bad = _the_ok(toks)
        c = _vfrac(toks, vocab_full)
        cov[i] = a
        feats[i] = (a, n, b, bad, c)

    nt = len(target_sample)
    if nt < 20 or not bg_full:
        return list(keep), cov

    halves = [[j for j in range(nt) if j % 2 == h] for h in (0, 1)]
    half_sets = [_build_sets([full_tok[j] for j in halves[h]]) for h in (0, 1)]
    a_vals, b_vals, c_vals = [], [], []
    for h in (0, 1):
        bg_h, voc_h = half_sets[h]
        for j in halves[1 - h]:
            toks = _toks(target_sample[j])
            if len(toks) < 2:
                continue
            a_vals.append(_cov(toks, bg_h)[0])
            b_vals.append(_the_ok(toks)[0])
            c_vals.append(_vfrac(toks, voc_h))
    if len(a_vals) < 10:
        return list(keep), cov

    r_a, r_c = 1.0, 1.0
    sample = rng.sample(list(keep), min(N_CALIB_POOL, len(keep)))
    if sample:
        s_cov = np.array([cov[i] for i in sample])
        q75 = float(np.percentile(s_cov, 75))
        ra_list, rc_list = [], []
        for i in sample:
            if cov[i] < q75 or cov[i] <= 0:
                continue
            toks = _toks(docs[i])
            c_full = feats[i][4]
            for h in (0, 1):
                bg_h, voc_h = half_sets[h]
                ah = _cov(toks, bg_h)[0]
                ch = _vfrac(toks, voc_h)
                if ah > 0:
                    ra_list.append(cov[i] / ah)
                if ch > 0:
                    rc_list.append(c_full / ch)
        if ra_list:
            r_a = float(min(2.5, max(1.0, np.median(ra_list))))
        if rc_list:
            r_c = float(min(1.3, max(1.0, np.median(rc_list))))

    thr_a = float(np.percentile(a_vals, 2)) * r_a - MARGIN_A
    thr_b = float(np.percentile(b_vals, 2)) - MARGIN_B
    thr_c = min(1.0, float(np.percentile(c_vals, 2)) * r_c) - MARGIN_C

    ref = 1.0 / np.sqrt(1000.0)
    out = []
    for i in keep:
        a, n, b, bad, c = feats[i]
        if n <= 0:
            continue
        allow = max(0.0, 0.75 * (1.0 / np.sqrt(n) - ref))
        if a < thr_a - allow:
            continue
        if b < thr_b and bad >= 2:
            continue
        if c < thr_c:
            continue
        out.append(i)

    if len(out) < 20:
        return list(keep), cov
    return out, cov

def _norm_hash(tag, s):
    s = _WS.sub(" ", s.lower()).strip()
    return tag + hashlib.md5(s.encode("utf-8", "ignore")).digest()

def _doc_keys(d):
    keys = [_norm_hash(b"h", d[:500])]
    stripped = d.lstrip()
    first_line = stripped.split("\n", 1)[0].strip()
    m = _TITLE.match(first_line)
    if m:
        title = _WS.sub(" ", m.group(1).lower()).strip()
        if title:
            keys.append(_norm_hash(b"t", title))
    n = len(d)
    if n >= 1000:
        for tag, f in ((b"1", 0.25), (b"2", 0.5), (b"3", 0.75)):
            p = int(n * f)
            keys.append(_norm_hash(b"w" + tag, d[p:p + 200]))
    return keys

def _dedup(docs, keep, cov):
    order = sorted(keep, key=lambda i: (-cov.get(i, 0.0), -len(docs[i]), i))
    seen = set()
    out = []
    for i in order:
        keys = _doc_keys(docs[i])
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(i)
    out.sort()
    return out

def _score(docs, keep, target_sample, seed, rng, t0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def make_vec():
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=200000,
            sublinear_tf=True,
            dtype=np.float32,
        )

    nt = len(target_sample)
    m = len(keep)

    t_texts_full = [t[:PREFIX] for t in target_sample]
    c0 = time.time()
    vec = make_vec()
    vec.fit(t_texts_full)
    dt = max(time.time() - c0, 1e-3)
    rate = max(sum(len(t) for t in t_texts_full), 1) / dt
    remaining = TIME_LIMIT - (time.time() - t0)
    vec_time = max(10.0, min(55.0, 0.35 * remaining))
    budget = min(40e6, rate * vec_time)
    prefix = int(max(200, min(PREFIX, budget / max(m, 1))))

    texts = [t[:prefix] for t in target_sample] + [docs[i][:prefix] for i in keep]
    X = vec.transform(texts).tocsr()
    del texts
    
    Xt = X[:nt]
    Xp = X[nt:]
    
    # Compute centroid of target samples
    # TfidfVectorizer.transform returns a sparse matrix, we can mean across rows
    # However, for TF-IDF, we can't easily mean raw vectors; we sum the TF-IDF weights.
    # Since we need cosine similarity, we need to be careful about normalization.
    # The easiest robust way: compute mean of the TF-IDF vectors.
    # Because TfidfVectorizer's transform already applies L2 normalization, 
    # we can't just average them to get a unit vector.
    # Instead, let's compute the cosine similarity to the mean of the raw TF scores
    # but for simplicity and speed, we average the normalized vectors and re-normalize.
    
    # A very common and fast approach for "centroid" cosine similarity:
    # Just average the sparse rows and normalize the result.
    centroid = Xt.mean(axis=0)
    # Normalize the centroid to unit length
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    # Calculate cosine similarity to centroid
    # Cosine similarity between unit vector centroid and Xp:
    # score = dot(centroid, Xp_i) / (||centroid|| * ||Xp_i||)
    # since ||centroid||=1, score = dot(centroid, Xp_i) / ||Xp_i||
    
    # Note: Xp.tocsr() dot product is efficient
    # We compute the dot product first
    dot_products = (Xp.dot(centroid.T)).toarray().flatten()
    
    # Compute L2 norms of each row in Xp
    # Xp.power(2).sum(axis=1) gives squared norms
    sq_norms = np.array(Xp.power(2).sum(axis=1)).flatten()
    # Avoid division by zero
    sq_norms[sq_norms == 0] = 1e-9
    pool_scores = dot_products / np.sqrt(sq_norms)

    # Final score with bonus
    pool_scores += np.array([_format_bonus(docs[i]) for i in keep])
    
    # Use a simple mean of target sample scores (similarity to centroid) as threshold
    # since all target docs are roughly at the centroid.
    # The dot product of Xt with the normalized centroid gives the similarity of target docs to the centroid.
    target_dots = (Xt.dot(centroid.T)).toarray().flatten()
    target_sq_norms = np.array(Xt.power(2).sum(axis=1)).flatten()
    target_sq_norms[target_sq_norms == 0] = 1e-9
    target_scores = target_dots / np.sqrt(target_sq_norms)

    # Threshold is mean of similarities
    thr = 0.5 * (float(np.mean(target_scores)) + float(np.mean(pool_scores)))
    return pool_scores, thr

def _count_until(docs, ranked, goal, t0):
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    cum = []
    total = 0
    chars_seen, toks_seen = 0, 0
    chunk = 1000
    pos = 0
    n = len(ranked)
    while pos < n and total < goal:
        ids = ranked[pos:pos + chunk]
        batch = [docs[i] for i in ids]
        counts = None
        if enc is not None and (time.time() - t0) < 0.85 * TIME_LIMIT:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch(batch, num_threads=8)]
                chars
