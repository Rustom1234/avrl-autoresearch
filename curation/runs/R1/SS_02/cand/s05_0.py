"""Curator: keep only clean, target-like documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

1. Word-order score h: fraction of a document's first ~400 adjacent lowercase
   word bigrams that occur in the target sample.
2. Domain score p (first pass): TF-IDF (word 1-2 grams) + logistic regression,
   with the target sample as positives and 3000 seeded-random pool documents as
   negatives.
3. Self-training round: the first-pass negatives are contaminated with
   target-like documents, so the classifier is refit once.
   Positives: the target sample plus up to 3000 seeded-random pool documents
   that pass the first-pass filter (h > h_thr and p >= 0.5, after dedup).
   Negatives: 4000 seeded-random pool documents that confidently fail
   (h <= h_thr or p < 0.2). This gives the second-pass score p2.
4. Candidates are documents with h above a threshold calibrated leave-one-out
   on the target sample and p2 >= 0.5, dropping short documents.
5. Whole-document damage filters, computed only for the candidates and applied
   before dedup (so a damaged copy cannot shadow its clean twin):
   - h_min: minimum over consecutive 400-bigram windows of the full text of
     the fraction of bigrams found in the target sample.
   - case-sensitive surface statistics over the full text: token coverage by
     the target vocabulary, uppercase letter ratio, share of ',' and '.' that
     stand as separate tokens, and share of characters never seen in the
     target sample.
   All thresholds are calibrated on the target sample. If fewer than 50
   documents survive, the selection without these filters is used instead.
6. (Near-)duplicates are dropped.

If the kept set holds fewer than T tokens it is returned as is (the trainer
cycles it). If it holds more, the highest-p2 documents up to T tokens are kept.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

N_BIGRAMS = 400        # bigrams scored per document (head score) / per window
MIN_LAST_WINDOW = 200  # a shorter final window is merged into the previous one
HEAD_CHARS = 6000      # characters scanned to get the first ~400 words
CLF_CHARS = 3000       # characters per document seen by the classifier
N_NEG = 3000           # random pool documents used as first-pass negatives
N_POS2 = 3000          # max pool documents added as second-pass positives
N_NEG2 = 4000          # confidently failing pool documents used as second-pass negatives
MIN_CHARS = 200        # shorter documents are dropped
DEDUP_CHARS = 300      # prefix length for the near-duplicate key
P_MIN = 0.5
P_NEG_MAX = 0.2        # p below this counts as a confident fail
MIN_SURVIVORS = 50     # fewer clean documents than this -> skip damage filters
ODD_FLOOR = 0.005      # minimum allowed threshold for the odd-character share


def _head_bigrams(text):
    """The first N_BIGRAMS adjacent lowercase word bigrams of text."""
    w = text[:HEAD_CHARS].lower().split()[: N_BIGRAMS + 1]
    return list(zip(w, w[1:]))


def _all_bigrams(text):
    w = text.lower().split()
    return set(zip(w, w[1:]))


def _count_tokens(texts):
    enc = tiktoken.get_encoding("gpt2")
    out = []
    for s in range(0, len(texts), 2000):
        chunk = texts[s:s + 2000]
        try:
            out.extend(len(t) for t in enc.encode_ordinary_batch(chunk, num_threads=8))
        except Exception:
            out.extend(len(enc.encode_ordinary(t)) for t in chunk)
    return out


def _fit_and_score(pos_txt, neg_txt, docs, seed):
    """Fit TF-IDF + logistic regression and return P(target) for every doc."""
    n = len(docs)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )
    X_train = vec.fit_transform(pos_txt + neg_txt)
    y_train = np.array([1] * len(pos_txt) + [0] * len(neg_txt))
    # Balanced weights: the two classes can differ a lot in size.
    clf = LogisticRegression(
        C=10.0, max_iter=1000, class_weight="balanced", random_state=seed
    )
    clf.fit(X_train, y_train)

    p = np.zeros(n, dtype=np.float64)
    step = 5000
    for s in range(0, n, step):
        X = vec.transform([d[:CLF_CHARS] for d in docs[s:s + step]])
        p[s:s + step] = clf.predict_proba(X)[:, 1]
    return p


def _passing(docs, h, h_thr, p, clean=None):
    """Indices of documents that pass the filter, after MIN_CHARS and dedup.

    If clean is given (a set of indices), documents outside it are dropped
    before dedup.
    """
    kept = []
    seen_full = set()
    seen_head = set()
    for i in range(len(docs)):
        d = docs[i]
        if len(d) < MIN_CHARS:
            continue
        if not (h[i] > h_thr and p[i] >= P_MIN):
            continue
        if clean is not None and i not in clean:
            continue
        # Dedup only among documents that pass, so a damaged copy cannot
        # shadow its clean twin.
        full_key = hash(d)
        head_key = " ".join(d[:DEDUP_CHARS].lower().split())
        if full_key in seen_full or head_key in seen_head:
            continue
        seen_full.add(full_key)
        seen_head.add(head_key)
        kept.append(i)
    return kept


def _window_min(flags):
    """Minimum over consecutive windows of the mean of flags (list of 0/1).

    Windows hold N_BIGRAMS entries; a final window shorter than
    MIN_LAST_WINDOW is merged into the previous one. Returns None if empty.
    """
    m = len(flags)
    if m == 0:
        return None
    bounds = list(range(0, m, N_BIGRAMS)) + [m]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_LAST_WINDOW:
        del bounds[-2]
    best = 1.0
    for a, b in zip(bounds, bounds[1:]):
        f = sum(flags[a:b]) / (b - a)
        if f < best:
            best = f
    return best


def _surface(text, tok_ok, char_ok):
    """Case-sensitive surface statistics of the full text.

    Returns (cov, up, punct, odd); cov is None if there are no tokens and
    punct is None if the text holds no ',' or '.' characters.
    """
    toks = text.split()
    if toks:
        cov = sum(1 for w in toks if tok_ok(w)) / len(toks)
    else:
        cov = None

    cc = Counter(text)
    letters = 0
    upper = 0
    odd = 0
    for ch, c in cc.items():
        if ch.isalpha():
            letters += c
            if ch.isupper():
                upper += c
        if ch.isascii() and ch.isalnum():
            continue
        if ch.isspace():
            continue
        if not char_ok(ch):
            odd += c
    up = upper / letters if letters else 0.0
    odd_frac = odd / len(text) if text else 0.0

    n_punct_chars = cc[","] + cc["."]
    if n_punct_chars:
        punct = (toks.count(",") + toks.count(".")) / n_punct_chars
    else:
        punct = None
    return cov, up, punct, odd_frac


def _pct(values, q):
    """Percentile of the non-None values, or None if there are none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.percentile(vals, q))


def _clean_set(docs, cand, target_sample, df, target_bigrams):
    """Indices in cand that pass every whole-document damage test."""
    # ----- calibration on the target sample (leave-one-out) -----
    df_tok = Counter()
    df_chr = Counter()
    for t in target_sample:
        df_tok.update(set(t.split()))
        df_chr.update(set(t))
    vocab = set(df_tok)
    target_chars = set(df_chr)

    t_hmin, t_cov, t_up, t_punct, t_odd = [], [], [], [], []
    for t in target_sample:
        w = t.lower().split()
        flags = [1 if df[b] >= 2 else 0 for b in zip(w, w[1:])]
        t_hmin.append(_window_min(flags))
        # Leave-one-out: a token / character counts as known only if another
        # target document also contains it (the document itself accounts for
        # one occurrence in df).
        cov, up, punct, odd = _surface(
            t,
            lambda x: df_tok[x] >= 2,
            lambda ch: df_chr[ch] >= 2,
        )
        t_cov.append(cov)
        t_up.append(up)
        t_punct.append(punct)
        t_odd.append(odd)

    q = _pct(t_hmin, 5)
    hmin_thr = 0.9 * q if q is not None else None
    q = _pct(t_cov, 5)
    cov_thr = 0.9 * q if q is not None else None
    q = _pct(t_up, 99)
    up_thr = 1.5 * q if q is not None else None
    q = _pct(t_punct, 2)
    punct_thr = 0.9 * q if q is not None else None
    q = _pct(t_odd, 99)
    odd_thr = max(ODD_FLOOR, 2.0 * q) if q is not None else ODD_FLOOR

    in_vocab = vocab.__contains__
    in_chars = target_chars.__contains__

    clean = set()
    for i in cand:
        d = docs[i]

        # (1) windowed word-order score
        if hmin_thr is not None:
            w = d.lower().split()
            flags = [1 if b in target_bigrams else 0 for b in zip(w, w[1:])]
            h_min = _window_min(flags)
            if h_min is None or not (h_min > hmin_thr):
                continue

        # (2) surface statistics
        cov, up, punct, odd = _surface(d, in_vocab, in_chars)
        if cov_thr is not None:
            if cov is None or not (cov > cov_thr):
                continue
        if up_thr is not None and not (up < up_thr):
            continue
        # A document without any ',' or '.' gives no evidence about
        # punctuation spacing, so the test is skipped for it.
        if punct_thr is not None and punct is not None and not (punct > punct_thr):
            continue
        if not (odd <= odd_thr):
            continue

        clean.add(i)
    return clean


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def fallback():
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    if n == 0:
        return []
    if not target_sample:
        return fallback()

    # ---------- 1. word-order score ----------
    target_sets = [_all_bigrams(t) for t in target_sample]
    df = Counter()
    for s in target_sets:
        df.update(s)
    target_bigrams = set(df)

    h = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(docs):
        bg = _head_bigrams(d)
        if bg:
            hits = 0
            for b in bg:
                if b in target_bigrams:
                    hits += 1
            h[i] = hits / len(bg)

    # Leave-one-out h of each target document against the other 199.
    # A bigram of document i is in "the others" iff it occurs in >= 2 target docs
    # (document i itself always accounts for one).
    loo = []
    for t in target_sample:
        bg = _head_bigrams(t)
        if not bg:
            continue
        loo.append(sum(1 for b in bg if df[b] >= 2) / len(bg))
    h_thr = 0.9 * float(np.percentile(loo, 5)) if loo else 0.0

    # ---------- 2. domain score, first pass ----------
    target_txt = [t[:CLF_CHARS] for t in target_sample]
    neg_idx = rng.sample(range(n), min(N_NEG, n))
    neg_txt = [docs[i][:CLF_CHARS] for i in neg_idx]
    p = _fit_and_score(target_txt, neg_txt, docs, seed)

    # ---------- 3. self-training round ----------
    # Positives: target sample + pool documents that pass the current filter.
    # Negatives: pool documents that confidently fail, so no target-like
    # document sits in the negatives.
    pass1 = _passing(docs, h, h_thr, p)
    fail1 = [i for i in range(n) if h[i] <= h_thr or p[i] < P_NEG_MAX]

    pos2_idx = rng.sample(pass1, min(N_POS2, len(pass1)))
    neg2_idx = rng.sample(fail1, min(N_NEG2, len(fail1)))

    if neg2_idx:
        pos2_txt = target_txt + [docs[i][:CLF_CHARS] for i in pos2_idx]
        neg2_txt = [docs[i][:CLF_CHARS] for i in neg2_idx]
        p2 = _fit_and_score(pos2_txt, neg2_txt, docs, seed)
    else:
        # No confident negatives to train on: keep the first-pass score.
        p2 = p

    # ---------- 4. whole-document damage filters ----------
    # Computed only for documents that already pass h and p2 (and MIN_CHARS),
    # and applied before dedup.
    cand = [
        i for i in range(n)
        if len(docs[i]) >= MIN_CHARS and h[i] > h_thr and p2[i] >= P_MIN
    ]
    clean = _clean_set(docs, cand, target_sample, df, target_bigrams)

    # ---------- 5. select ----------
    kept = _passing(docs, h, h_thr, p2, clean=clean)
    if len(kept) < MIN_SURVIVORS:
        # Too few clean documents: use the selection without damage filters.
        kept = _passing(docs, h, h_thr, p2)

    if not kept:
        return fallback()

    ntok = _count_tokens([docs[i] for i in kept])
    total = sum(ntok)

    if total > T:
        order = sorted(range(len(kept)), key=lambda j: (-p2[kept[j]], kept[j]))
        chosen = []
        acc = 0
        for j in order:
            chosen.append(kept[j])
            acc += ntok[j]
            if acc >= T:
                break
        kept = sorted(chosen)

    rng.shuffle(kept)
    return kept
