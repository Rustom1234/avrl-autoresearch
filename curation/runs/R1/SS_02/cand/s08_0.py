"""Curator: keep only clean, intact, target-like documents.

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
4. Candidates: h > h_thr, p2 >= 0.1 and at least MIN_CHARS characters.
5. Whole-document format / intactness gate. Each test is computed over the
   full text and calibrated on the target sample:
   (a) h_min: minimum over consecutive 400-bigram windows of the fraction of
       lowercase word bigrams found in the target bigram set.
   (b) cov: fraction of case-sensitive whitespace tokens found in the target
       token set.
   (c) up: uppercase letters / all letters, bounded above and below (the
       lower bound drops lowercased copies).
   (d) punct: tokens equal to ',' or '.' / count of ',' and '.' characters.
   (e) odd: fraction of characters that are not ASCII alphanumerics,
       whitespace, or any character seen in the target sample.
   (f) dig: ASCII digits / all characters, bounded above (drops table and
       number dumps).
   (g) tpc: GPT-2 tokens per character, bounded on both sides. Typos,
       character swaps, missing spaces, wrong case or detokenised punctuation
       raise it; repetitive or boilerplate text lowers it.
6. Documents that pass all tests are deduplicated (after the tests, so a
   damaged copy cannot shadow its clean twin).
7. The output is built to about T tokens using quality q = p2 + h_min + cov:
   if the survivors exceed T tokens the highest-q documents up to T are kept;
   if they fall short every survivor is included once and further copies are
   added in descending q until T tokens are reached.
8. If fewer than 50 documents survive, the previous selection rule is used
   (h > h_thr and p2 >= 0.5, dedup, highest p2 up to T tokens).

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

N_BIGRAMS = 400        # bigrams scored per document
HEAD_CHARS = 6000      # characters scanned to get the first ~400 words
CLF_CHARS = 3000       # characters per document seen by the classifier
N_NEG = 3000           # random pool documents used as first-pass negatives
N_POS2 = 3000          # max pool documents added as second-pass positives
N_NEG2 = 4000          # confidently failing pool documents used as second-pass negatives
MIN_CHARS = 200        # shorter documents are dropped
DEDUP_CHARS = 300      # prefix length for the near-duplicate key
P_MIN = 0.5            # classifier threshold for self-training and the fallback
P_NEG_MAX = 0.2        # p below this counts as a confident fail
P_CAND = 0.1           # classifier threshold for final candidates
WIN = 400              # bigrams per window for h_min
WIN_MIN = 200          # a final window shorter than this is merged into the previous one
MIN_SURVIVORS = 50     # below this, fall back to the previous selection
UP_LO_FACTOR = 0.7     # lower case bound = factor * 1st percentile of target up
TPC_LO_FACTOR = 0.93   # lower tpc bound = factor * 1st percentile of target tpc
TPC_HI_FACTOR = 1.07   # upper tpc bound = factor * 99th percentile of target tpc
DIG_FLOOR = 0.02       # the digit-fraction threshold is never below this
DIG_FACTOR = 1.5       # digit threshold = factor * 99th percentile of target dig

_DIGITS = "0123456789"


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


def _dedup(docs, indices):
    """Drop exact and near duplicates (same normalised prefix), keeping the first."""
    kept = []
    seen_full = set()
    seen_head = set()
    for i in indices:
        d = docs[i]
        full_key = hash(d)
        head_key = " ".join(d[:DEDUP_CHARS].lower().split())
        if full_key in seen_full or head_key in seen_head:
            continue
        seen_full.add(full_key)
        seen_head.add(head_key)
        kept.append(i)
    return kept


def _passing(docs, h, h_thr, p):
    """Indices of documents that pass the filter, after MIN_CHARS and dedup."""
    # Dedup only among documents that pass, so a damaged copy cannot
    # shadow its clean twin.
    ok = [
        i for i in range(len(docs))
        if len(docs[i]) >= MIN_CHARS and h[i] > h_thr and p[i] >= P_MIN
    ]
    return _dedup(docs, ok)


def _window_min(flags):
    """Minimum hit fraction over consecutive windows of WIN flags.

    A final window shorter than WIN_MIN is merged into the previous one.
    """
    m = len(flags)
    if m == 0:
        return 0.0
    nwin = m // WIN
    edges = list(range(0, nwin * WIN + 1, WIN))
    rem = m - nwin * WIN
    if rem > 0:
        if rem >= WIN_MIN or nwin == 0:
            edges.append(m)
        else:
            edges[-1] = m
    best = 1.0
    for a, b in zip(edges, edges[1:]):
        f = sum(flags[a:b]) / (b - a)
        if f < best:
            best = f
    return best


def _char_stats(text, target_chars):
    """(up, n_punct_chars, odd, dig) for a text, from its character counts."""
    cnt = Counter(text)
    letters = 0
    upper = 0
    odd = 0
    for c, v in cnt.items():
        if c.isalpha():
            letters += v
            if c.isupper():
                upper += v
        if c in target_chars or c.isspace() or (c.isascii() and c.isalnum()):
            continue
        odd += v
    up = upper / letters if letters else 1.0
    n_punct_chars = cnt.get(",", 0) + cnt.get(".", 0)
    n_digits = sum(cnt.get(c, 0) for c in _DIGITS)
    if text:
        return up, n_punct_chars, odd / len(text), n_digits / len(text)
    return up, n_punct_chars, 1.0, 1.0


def _punct(tokens, n_punct_chars):
    if n_punct_chars == 0:
        return 0.0
    return (tokens.count(",") + tokens.count(".")) / n_punct_chars


def _pct(values, q, default):
    return float(np.percentile(values, q)) if values else default


def _legacy_selection(docs, h, h_thr, p2):
    """The previous selection: h > h_thr and p2 >= P_MIN, dedup, top p2 up to T."""
    kept = _passing(docs, h, h_thr, p2)
    if not kept:
        return []
    ntok = _count_tokens([docs[i] for i in kept])
    if sum(ntok) > T:
        order = sorted(range(len(kept)), key=lambda j: (-p2[kept[j]], kept[j]))
        chosen = []
        acc = 0
        for j in order:
            chosen.append(kept[j])
            acc += ntok[j]
            if acc >= T:
                break
        kept = sorted(chosen)
    return kept


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

    # ---------- 4. calibrate the whole-document tests on the target ----------
    df_tok = Counter()
    for t in target_sample:
        df_tok.update(set(t.split()))
    target_tokens = set(df_tok)
    target_chars = set()
    for t in target_sample:
        target_chars.update(t)

    t_hmin, t_cov, t_up, t_punct, t_odd, t_dig = [], [], [], [], [], []
    for t in target_sample:
        if not t:
            continue
        w = t.lower().split()
        flags = [df[b] >= 2 for b in zip(w, w[1:])]
        if flags:
            t_hmin.append(_window_min(flags))
        toks = t.split()
        if toks:
            t_cov.append(sum(1 for x in toks if df_tok[x] >= 2) / len(toks))
        up, n_pc, odd, dig = _char_stats(t, target_chars)
        t_up.append(up)
        t_punct.append(_punct(toks, n_pc))
        t_odd.append(odd)
        t_dig.append(dig)

    # Tokens per character of the target documents.
    target_nonempty = [t for t in target_sample if t]
    t_tpc = [
        k / len(t)
        for k, t in zip(_count_tokens(target_nonempty), target_nonempty)
    ]

    hmin_thr = 0.9 * _pct(t_hmin, 5, 0.0)
    cov_thr = 0.9 * _pct(t_cov, 5, 0.0)
    up_thr = 1.5 * _pct(t_up, 99, 1.0)
    up_lo = UP_LO_FACTOR * _pct(t_up, 1, 0.0)
    punct_thr = 0.9 * _pct(t_punct, 2, 0.0)
    odd_thr = max(0.005, 2.0 * _pct(t_odd, 99, 0.0))
    dig_thr = max(DIG_FLOOR, DIG_FACTOR * _pct(t_dig, 99, 0.0))
    tpc_lo = TPC_LO_FACTOR * _pct(t_tpc, 1, 0.0)
    tpc_hi = TPC_HI_FACTOR * _pct(t_tpc, 99, float("inf"))

    # ---------- 5. candidates and whole-document gate ----------
    in_bigrams = target_bigrams.__contains__
    in_tokens = target_tokens.__contains__
    passed5 = []
    quality = {}
    for i in range(n):
        d = docs[i]
        if len(d) < MIN_CHARS:
            continue
        if not (h[i] > h_thr and p2[i] >= P_CAND):
            continue

        # (c), (d), (e), (f): character-level tests
        up, n_pc, odd, dig = _char_stats(d, target_chars)
        if not (up < up_thr):
            continue
        # Lower bound on case: drops lowercased copies. If the target itself
        # has (almost) no uppercase the bound is 0 and nothing is dropped.
        if up_lo > 0.0 and not (up > up_lo):
            continue
        if not (odd <= odd_thr):
            continue
        if not (dig <= dig_thr):
            continue
        toks = d.split()
        if not toks:
            continue
        if not (_punct(toks, n_pc) > punct_thr):
            continue

        # (b): token coverage
        cov = sum(map(in_tokens, toks)) / len(toks)
        if not (cov > cov_thr):
            continue

        # (a): worst-window word-order score
        w = d.lower().split()
        flags = list(map(in_bigrams, zip(w, w[1:])))
        h_min = _window_min(flags)
        if not (h_min > hmin_thr):
            continue

        passed5.append(i)
        quality[i] = float(p2[i]) + h_min + cov

    # (g): GPT-2 tokens per character. Counted once here, before dedup, and
    # reused for the build step.
    ntok_of = {}
    passed = []
    if passed5:
        counts = _count_tokens([docs[i] for i in passed5])
        for i, k in zip(passed5, counts):
            tpc = k / len(docs[i])
            if tpc_lo <= tpc <= tpc_hi:
                passed.append(i)
                ntok_of[i] = k

    # Dedup after the tests, so a damaged copy cannot shadow its clean twin.
    survivors = _dedup(docs, passed)

    # ---------- 6. fallback to the previous selection ----------
    if len(survivors) < MIN_SURVIVORS:
        kept = _legacy_selection(docs, h, h_thr, p2)
        if not kept:
            return fallback()
        rng.shuffle(kept)
        return kept

    # ---------- 7. build the output to about T tokens ----------
    ntok = [ntok_of[i] for i in survivors]
    total = sum(ntok)
    order = sorted(
        range(len(survivors)), key=lambda j: (-quality[survivors[j]], survivors[j])
    )

    if total > T:
        out = []
        acc = 0
        for j in order:
            out.append(survivors[j])
            acc += ntok[j]
            if acc >= T:
                break
    else:
        out = list(survivors)
        acc = total
        while 0 < total and acc < T:
            for j in order:
                out.append(survivors[j])
                acc += ntok[j]
                if acc >= T:
                    break

    out.sort()
    rng.shuffle(out)
    return out
