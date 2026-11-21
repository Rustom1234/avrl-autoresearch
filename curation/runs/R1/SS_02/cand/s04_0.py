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
4. Surface-format filter (final selection only): for documents with
   h > h_thr and p2 >= 0.5, four case-sensitive statistics are computed over
   the FULL text:
     cov   - fraction of whitespace tokens present in the target vocabulary;
     up    - uppercase letters / all letters;
     punct - fraction of ',' and '.' characters that are stand-alone tokens;
     odd   - fraction of characters that are neither ASCII alphanumerics,
             whitespace, nor any character seen in the target sample.
   Thresholds are calibrated on the target sample (cov leave-one-out).
   Documents failing any test are dropped before dedup, so a damaged copy
   cannot shadow its clean twin. If fewer than 50 documents survive, the
   format filter is skipped.
5. Keep the surviving documents, dropping short documents and
   (near-)duplicates.

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

N_BIGRAMS = 400        # bigrams scored per document
HEAD_CHARS = 6000      # characters scanned to get the first ~400 words
CLF_CHARS = 3000       # characters per document seen by the classifier
N_NEG = 3000           # random pool documents used as first-pass negatives
N_POS2 = 3000          # max pool documents added as second-pass positives
N_NEG2 = 4000          # confidently failing pool documents used as second-pass negatives
MIN_CHARS = 200        # shorter documents are dropped
DEDUP_CHARS = 300      # prefix length for the near-duplicate key
P_MIN = 0.5
P_NEG_MAX = 0.2        # p below this counts as a confident fail
MIN_FORMAT_KEPT = 50   # below this the surface-format filter is skipped
ODD_FLOOR = 0.005      # minimum allowed threshold for the odd-character rate


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


def _passing(docs, h, h_thr, p, ok=None):
    """Indices of documents that pass the filter, after MIN_CHARS and dedup.

    ok: optional boolean mask; documents with ok[i] False are dropped
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
        if ok is not None and not ok[i]:
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


class _SurfaceStats:
    """Case-sensitive whole-document surface statistics."""

    def __init__(self, target_chars):
        self._target_chars = target_chars
        self._cache = {}   # char -> (is_letter, is_upper, is_odd)

    def _char_class(self, c):
        cls = self._cache.get(c)
        if cls is None:
            is_letter = c.isalpha()
            is_upper = is_letter and c.isupper()
            is_odd = not (
                (c.isascii() and c.isalnum())
                or c.isspace()
                or c in self._target_chars
            )
            cls = (is_letter, is_upper, is_odd)
            self._cache[c] = cls
        return cls

    def stats(self, text, vocab):
        """Return (cov, up, punct, odd) for text.

        punct is None when the text has no ',' or '.' characters.
        """
        tokens = text.split()
        if tokens:
            cov = sum(map(vocab.__contains__, tokens)) / len(tokens)
        else:
            cov = 0.0

        counts = Counter(text)
        letters = 0
        uppers = 0
        odd_n = 0
        for c, k in counts.items():
            is_letter, is_upper, is_odd = self._char_class(c)
            if is_letter:
                letters += k
                if is_upper:
                    uppers += k
            if is_odd:
                odd_n += k
        up = uppers / letters if letters else 0.0
        odd = odd_n / len(text) if text else 0.0

        punct_chars = counts.get(",", 0) + counts.get(".", 0)
        if punct_chars:
            punct = (tokens.count(",") + tokens.count(".")) / punct_chars
        else:
            punct = None
        return cov, up, punct, odd


def _format_mask(docs, target_sample, candidates):
    """Boolean mask over docs: True where the surface-format tests pass.

    Only the indices in candidates are evaluated; all others stay False.
    Returns None if the tests cannot be calibrated.
    """
    # Target vocabulary (case-sensitive whitespace tokens) and document frequency.
    df_tok = Counter()
    target_chars = set()
    for t in target_sample:
        df_tok.update(set(t.split()))
        target_chars.update(t)
    vocab = set(df_tok)
    # Leave-one-out vocabulary: a token of document i is in "the others"
    # iff it occurs in >= 2 target documents.
    vocab_loo = {tok for tok, c in df_tok.items() if c >= 2}

    ss = _SurfaceStats(target_chars)

    t_cov, t_up, t_punct, t_odd = [], [], [], []
    for t in target_sample:
        if not t.split():
            continue
        cov, up, punct, odd = ss.stats(t, vocab_loo)
        t_cov.append(cov)
        t_up.append(up)
        t_odd.append(odd)
        if punct is not None:
            t_punct.append(punct)
    if not t_cov:
        return None

    cov_thr = 0.9 * float(np.percentile(t_cov, 5))
    up_thr = 1.5 * float(np.percentile(t_up, 99))
    punct_thr = 0.9 * float(np.percentile(t_punct, 2)) if t_punct else 0.0
    odd_thr = max(ODD_FLOOR, 2.0 * float(np.percentile(t_odd, 99)))

    ok = np.zeros(len(docs), dtype=bool)
    for i in candidates:
        cov, up, punct, odd = ss.stats(docs[i], vocab)
        if not cov > cov_thr:
            continue
        if not up < up_thr:
            continue
        # The punctuation test is only meaningful if the target actually uses
        # stand-alone ',' / '.' tokens. A document with no ',' or '.' at all
        # has no stand-alone punctuation and fails when the test is active.
        if punct_thr > 0.0:
            if punct is None or not punct > punct_thr:
                continue
        if not odd <= odd_thr:
            continue
        ok[i] = True
    return ok


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

    # ---------- 4. surface-format filter (final selection only) ----------
    candidates = [
        i for i in range(n)
        if len(docs[i]) >= MIN_CHARS and h[i] > h_thr and p2[i] >= P_MIN
    ]
    kept = None
    try:
        ok = _format_mask(docs, target_sample, candidates)
    except Exception:
        ok = None
    if ok is not None:
        kept_fmt = _passing(docs, h, h_thr, p2, ok=ok)
        if len(kept_fmt) >= MIN_FORMAT_KEPT:
            kept = kept_fmt

    # ---------- 5. select ----------
    if kept is None:
        # Too few survivors (or no calibration): use the unfiltered selection.
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
