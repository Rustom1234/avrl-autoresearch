"""Curator: keep only clean, intact, target-like documents, easiest-to-drop first.

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
4. Candidates: every document with at least MIN_CHARS characters. (h and p2
   are no longer used as a pre-filter; they only feed self-training, the
   quality score and the fallback.)
5. Whole-document format / intactness gate. Each test is computed over the
   full text and calibrated on the target sample. Tests run cheap-first with
   early exit:
   (c) up: uppercase letters / all letters.
   (e) odd: fraction of characters that are not ASCII alphanumerics,
       whitespace, or any character seen in the target sample.
   (d) punct: tokens equal to ',' or '.' / count of ',' and '.' characters.
   (b) cov: fraction of case-sensitive whitespace tokens found in the target
       token set.
   (a) h_min: minimum over consecutive 400-bigram windows of the fraction of
       lowercase word bigrams found in the target bigram set.
6. Documents that pass all five tests are deduplicated (after the tests, so a
   damaged copy cannot shadow its clean twin), longest copy first, so the
   longest copy is the one kept.
7. The output multiset is built to about T tokens using quality
   q = p2 + h_min + cov: if the survivors exceed T tokens the highest-q
   documents up to T are kept; if they fall short every survivor is included
   once and further copies are added in descending q until T tokens are
   reached.
8. Order (curriculum), two blocks, returned as A + B:
   Block A (read first): one copy each of the lowest-quality output
   documents until they make up 35% of the output tokens. Documents with
   p2 < 0.5 are taken first (ascending q), then the rest in ascending q, so
   every p2 < 0.5 survivor that fits is in block A. Shuffled with the seed.
   Block B (read last): all remaining output documents, including the extra
   copies, shuffled with the seed.
   Block A never holds more than one copy of a document, so a repeated
   document that reaches block A has its other copies in block B.
9. If fewer than 50 documents survive, the previous selection rule is used
   (h > h_thr and p2 >= 0.5, dedup, highest p2 up to T tokens, shuffled).

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import sys
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
P_MIN = 0.5            # classifier threshold for self-training, the fallback and block A priority
P_NEG_MAX = 0.2        # p below this counts as a confident fail
P_LOG = 0.1            # survivors with p2 below this are counted in the log
WIN = 400              # bigrams per window for h_min
WIN_MIN = 200          # a final window shorter than this is merged into the previous one
MIN_SURVIVORS = 50     # below this, fall back to the previous selection
A_FRAC = 0.35          # share of output tokens in block A (read first, lowest quality)


def _log(msg):
    print("[curate] " + msg, file=sys.stderr, flush=True)


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
    """(up, n_punct_chars, odd) for a text, from its character counts."""
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
    return up, n_punct_chars, (odd / len(text) if text else 1.0)


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

    t_hmin, t_cov, t_up, t_punct, t_odd = [], [], [], [], []
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
        up, n_pc, odd = _char_stats(t, target_chars)
        t_up.append(up)
        t_punct.append(_punct(toks, n_pc))
        t_odd.append(odd)

    hmin_thr = 0.9 * _pct(t_hmin, 5, 0.0)
    cov_thr = 0.9 * _pct(t_cov, 5, 0.0)
    up_thr = 1.5 * _pct(t_up, 99, 1.0)
    punct_thr = 0.9 * _pct(t_punct, 2, 0.0)
    odd_thr = max(0.005, 2.0 * _pct(t_odd, 99, 0.0))

    # ---------- 5. whole-document gate on every long-enough document ----------
    # Cheap-first with early exit: up, odd, punct, cov, then h_min.
    in_bigrams = target_bigrams.__contains__
    in_tokens = target_tokens.__contains__
    passed = []
    quality = {}
    for i in range(n):
        d = docs[i]
        if len(d) < MIN_CHARS:
            continue

        # (c), (e): character-level tests
        up, n_pc, odd = _char_stats(d, target_chars)
        if not (up < up_thr):
            continue
        if not (odd <= odd_thr):
            continue

        # (d): punctuation spacing (the only case-sensitive split of d)
        toks = d.split()
        if not toks:
            continue
        if not (_punct(toks, n_pc) > punct_thr):
            continue

        # (b): token coverage
        cov = sum(map(in_tokens, toks)) / len(toks)
        if not (cov > cov_thr):
            continue

        # (a): worst-window word-order score (lowercase split only here)
        w = d.lower().split()
        flags = list(map(in_bigrams, zip(w, w[1:])))
        h_min = _window_min(flags)
        if not (h_min > hmin_thr):
            continue

        passed.append(i)
        quality[i] = float(p2[i]) + h_min + cov

    # Dedup after the tests, so a damaged copy cannot shadow its clean twin.
    # Longest copy first, so the longest copy is the one kept.
    passed.sort(key=lambda i: (-len(docs[i]), i))
    survivors = _dedup(docs, passed)
    n_low = sum(1 for i in survivors if p2[i] < P_LOG)

    # ---------- 6. fallback to the previous selection ----------
    if len(survivors) < MIN_SURVIVORS:
        _log(
            "survivors=%d (p2<%.1f: %d) -> below %d, using legacy selection"
            % (len(survivors), P_LOG, n_low, MIN_SURVIVORS)
        )
        kept = _legacy_selection(docs, h, h_thr, p2)
        if not kept:
            return fallback()
        rng.shuffle(kept)
        return kept

    # ---------- 7. build the output multiset to about T tokens ----------
    ntok = _count_tokens([docs[i] for i in survivors])
    total = sum(ntok)
    _log(
        "survivors=%d tokens=%d (%.3f of T) p2<%.1f: %d"
        % (len(survivors), total, total / T if T else 0.0, P_LOG, n_low)
    )
    tok_of = {survivors[j]: ntok[j] for j in range(len(survivors))}
    order = sorted(
        range(len(survivors)), key=lambda j: (-quality[survivors[j]], survivors[j])
    )

    copies = Counter()
    if total > T:
        acc = 0
        for j in order:
            copies[survivors[j]] += 1
            acc += ntok[j]
            if acc >= T:
                break
    else:
        for i in survivors:
            copies[i] += 1
        acc = total
        while 0 < total and acc < T:
            for j in order:
                copies[survivors[j]] += 1
                acc += ntok[j]
                if acc >= T:
                    break

    # ---------- 8. two-block curriculum order ----------
    # Block A: one copy each of the lowest-quality output documents, up to
    # A_FRAC of the output tokens. p2 < P_MIN documents are taken first so that
    # every one that fits lands in block A; the rest follow in ascending q.
    # At most one copy per document goes to A, so other copies sit in B.
    budget = A_FRAC * acc
    order_a = sorted(
        copies, key=lambda i: (p2[i] >= P_MIN, quality[i], i)
    )
    block_a = []
    a_tok = 0
    for i in order_a:
        if a_tok >= budget:
            break
        block_a.append(i)
        a_tok += tok_of[i]
        copies[i] -= 1

    block_b = []
    for i in sorted(copies):
        c = copies[i]
        if c > 0:
            block_b.extend([i] * c)

    block_a.sort()
    rng.shuffle(block_a)
    rng.shuffle(block_b)
    _log(
        "block A: %d docs, %d tokens; block B: %d docs, %d tokens"
        % (len(block_a), a_tok, len(block_b), acc - a_tok)
    )
    return block_a + block_b
