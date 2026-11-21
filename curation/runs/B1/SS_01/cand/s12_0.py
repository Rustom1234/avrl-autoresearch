"""Curator: target-similarity filter plus junk / duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop exact duplicates (whitespace-normalised) and near-duplicates
     (same first 200 chars), keeping the longest copy.
  3. Drop damaged docs: heavy line repetition, very high compressibility,
     ' . ' mostly followed by a lowercase word (shuffled words).
  4. Word-order test for in-domain docs (score >= 0.5):
       a. old test (bigram hit rate against target_sample only, threshold
          0.5 x median of the top-scored docs) defines CLEAN0;
       b. a larger reference of word bigrams is built from target_sample
          plus the first 2000 words of the CLEAN0 docs at/above the CLEAN0
          median hit rate (bigram *document* counts);
       c. every deduped score >= 0.5 doc is re-scored against it on 5 evenly
          spaced 300-word windows (contributors need count >= 2, i.e. their
          own contribution is excluded; others need count >= 1);
       d. the drop threshold is the Otsu split of those hit fractions,
          clamped to [0.45, 0.85] x the median of the upper class.
     Dropped docs are never re-admitted as fill.
  5. Rank survivors by classifier score and take the top ones up to ~T tokens.
     If the in-domain docs fall short of T but reach 0.5T, return just those
     (the trainer cycles them); otherwise fill with next-best clean docs.
  6. Seeded shuffle of the final selection.
"""
import os
import random
import re
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_PERIOD_RE = re.compile(r" \. ([A-Za-z])")

_REF_WORDS = 2000      # words per contributor doc used for the reference
_WIN_WORDS = 300       # words per evaluation window
_N_WIN = 5             # evaluation windows per doc
_MIN_WORDS = 20        # below this a doc gets no hit fraction (never dropped)
_CLAMP_LO = 0.45
_CLAMP_HI = 0.85


def _scores(docs, target_sample, rng):
    """Probability-like target-domain score for every pool doc."""
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    # Prefix length: ~2000 chars, shrunk for very large pools to bound runtime.
    P = int(max(400, min(2000, 15e6 / max(n, 1))))

    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        lowercase=False,
        n_features=2 ** 18,
        alternate_sign=False,
        norm=None,
    )
    X_pool = hv.transform([d[:P] for d in docs])

    pos_chunks = []
    for t in target_sample:
        for k in range(5):
            c = t[k * P:(k + 1) * P]
            if not c.strip():
                break
            if k == 0 or len(c) >= P // 4:
                pos_chunks.append(c)
    if not pos_chunks:
        return np.full(n, 0.5)
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool)
    X_pos = tfidf.transform(X_pos)

    neg_idx = np.array(rng.sample(range(n), min(3000, n)), dtype=np.int64)

    def fit(neg):
        X = vstack([X_pos, X_pool[neg]]).tocsr()
        y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(len(neg))])
        clf = LogisticRegression(C=10.0, max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        return clf

    clf = fit(neg_idx)
    p = clf.predict_proba(X_pool)[:, 1]

    # PU refinement: the unlabeled "negatives" contain target-domain docs;
    # refit using only the reliably negative ones.
    reliable = neg_idx[p[neg_idx] < 0.3]
    if 100 <= len(reliable) < len(neg_idx):
        try:
            clf = fit(reliable)
            p = clf.predict_proba(X_pool)[:, 1]
        except Exception:
            pass
    return p


def _chars_per_token(docs, idxs, rng):
    """Calibrate chars/token with tiktoken on a small sample."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        sample = list(idxs)
        if len(sample) > 300:
            sample = rng.sample(sample, 300)
        chars = 0
        toks = 0
        for i in sample:
            s = docs[i][:5000]
            if not s:
                continue
            chars += len(s)
            toks += len(enc.encode(s, disallowed_special=()))
        if toks > 0 and chars > 0:
            return max(1.0, chars / toks)
    except Exception:
        pass
    return 4.0


# --------------------------------------------------------------------------
# word-bigram reference (bigrams packed as int64 keys: id_a << 32 | id_b)
# --------------------------------------------------------------------------
def _doc_keys_grow(words, vocab):
    """Unique bigram keys of a word list, adding unseen words to vocab."""
    if len(words) < 2:
        return None
    ids = np.fromiter(
        (vocab.setdefault(w, len(vocab)) for w in words),
        dtype=np.int64,
        count=len(words),
    )
    return np.unique((ids[:-1] << 32) | ids[1:])


def _build_reference(docs, target_sample, contributors):
    """Return (vocab, sorted keys, document counts) for the bigram reference."""
    vocab = {}
    parts = []
    for t in target_sample:
        k = _doc_keys_grow(t.lower().split(), vocab)
        if k is not None:
            parts.append(k)
    for i in contributors:
        k = _doc_keys_grow(docs[i].lower().split()[:_REF_WORDS], vocab)
        if k is not None:
            parts.append(k)
    if not parts:
        return vocab, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ref_keys, ref_counts = np.unique(np.concatenate(parts), return_counts=True)
    return vocab, ref_keys, ref_counts


def _hit_fraction(words, vocab, ref_keys, ref_counts, need):
    """Pooled bigram hit fraction over evenly spaced windows of a doc."""
    nw = len(words)
    if nw < _MIN_WORDS:
        return np.nan
    if nw <= _N_WIN * _WIN_WORDS:
        segs = [words]
    else:
        span = nw - _WIN_WORDS
        segs = []
        for j in range(_N_WIN):
            s = int(round(j * span / (_N_WIN - 1)))
            segs.append(words[s:s + _WIN_WORDS])
    hits = 0
    total = 0
    last = len(ref_keys) - 1
    get = vocab.get
    for seg in segs:
        if len(seg) < 2:
            continue
        ids = np.fromiter((get(w, -1) for w in seg), dtype=np.int64, count=len(seg))
        a = ids[:-1]
        b = ids[1:]
        total += len(a)
        ok = (a >= 0) & (b >= 0)
        if not ok.any():
            continue
        k = (a[ok] << 32) | b[ok]
        pos = np.searchsorted(ref_keys, k)
        pos[pos > last] = last
        found = ref_keys[pos] == k
        hits += int(np.count_nonzero(found & (ref_counts[pos] >= need)))
    if total <= 0:
        return np.nan
    return hits / total


def _otsu(values, bins=100):
    """Two-class variance-maximising threshold over a histogram of values."""
    values = np.asarray(values, dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    if not hi > lo:
        return None
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b = np.nan_to_num(sigma_b, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_b[-1] = 0.0
    k = int(np.argmax(sigma_b))
    if sigma_b[k] <= 0:
        return None
    return float(edges[k + 1])


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) classifier scores -------------------------------------------
    try:
        score = np.asarray(_scores(docs, target_sample, rng), dtype=np.float64)
    except Exception:
        score = np.full(n, 0.5)

    # ---- (2) duplicates ---------------------------------------------------
    dedup = np.ones(n, dtype=bool)
    seen_full = set()
    seen_head = set()
    order_by_len = sorted(range(n), key=lambda i: -len(docs[i]))
    for i in order_by_len:
        norm = " ".join(docs[i].split())
        if not norm:
            dedup[i] = False
            continue
        h_full = hash(norm)
        h_head = hash(norm[:200])
        if h_full in seen_full or h_head in seen_head:
            dedup[i] = False
            continue
        seen_full.add(h_full)
        seen_head.add(h_head)

    # ---- (3) damaged docs (non-bigram tests) + old bigram hit fraction ----
    target_bigrams = set()
    for t in target_sample:
        w = t.lower().split()
        target_bigrams.update(zip(w, w[1:]))

    other_ok = dedup.copy()          # survives dedup + line/compress/period tests
    bigram_frac = np.full(n, np.nan)  # old hit fraction (target_sample only)
    for i in range(n):
        if not other_ok[i]:
            continue
        d = docs[i]

        # heavy line repetition
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
            other_ok[i] = False
            continue

        # extreme compressibility
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500 and len(zlib.compress(raw, 1)) / len(raw) < 0.2:
            other_ok[i] = False
            continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                other_ok[i] = False
                continue

        # word-bigram hit fraction against the target sample
        if target_bigrams:
            w = d[:6000].lower().split()[:300]
            if len(w) >= 20:
                hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
                bigram_frac[i] = hits / (len(w) - 1)

    # old rule: fixed 0.5 x median of the top-scored docs
    old_bad = np.zeros(n, dtype=bool)
    valid = np.where(other_ok & ~np.isnan(bigram_frac))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_frac[top]))
        thr_old = 0.5 * med
        if thr_old > 0:
            old_bad = (
                other_ok & (score >= 0.5) & ~np.isnan(bigram_frac) & (bigram_frac < thr_old)
            )

    # ---- (4) sharpened word-order test -------------------------------------
    new_bad = None
    new_thr = float("nan")
    upper_med = float("nan")
    try:
        in_dom = score >= 0.5
        clean0 = other_ok & ~old_bad & in_dom
        c0 = np.where(clean0 & ~np.isnan(bigram_frac))[0]
        is_contrib = np.zeros(n, dtype=bool)
        if len(c0) > 0:
            c0_med = float(np.median(bigram_frac[c0]))
            is_contrib[c0[bigram_frac[c0] >= c0_med]] = True
        contributors = [int(i) for i in np.where(is_contrib)[0]]

        vocab, ref_keys, ref_counts = _build_reference(docs, target_sample, contributors)

        if len(ref_keys) > 0:
            new_frac = np.full(n, np.nan)
            for i in np.where(dedup & in_dom)[0]:
                i = int(i)
                need = 2 if is_contrib[i] else 1
                new_frac[i] = _hit_fraction(
                    docs[i].lower().split(), vocab, ref_keys, ref_counts, need
                )
            del vocab

            have = dedup & in_dom & ~np.isnan(new_frac)
            vals = new_frac[have]
            if len(vals) >= 20:
                otsu = _otsu(vals, bins=100)
                if otsu is not None:
                    upper = vals[vals >= otsu]
                    if len(upper) > 0:
                        upper_med = float(np.median(upper))
                        new_thr = float(
                            min(max(otsu, _CLAMP_LO * upper_med), _CLAMP_HI * upper_med)
                        )
                        if new_thr > 0:
                            new_bad = have & (new_frac < new_thr)
    except Exception as e:  # pragma: no cover
        print("[curate] new word-order test failed (%r); using old rule" % (e,), flush=True)
        new_bad = None

    used_new = new_bad is not None
    if used_new:
        keep = other_ok & ~new_bad      # dropped docs are never re-admitted as fill
    else:
        keep = other_ok & ~old_bad

    # ---- (5) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = surv[np.argsort(-score[surv], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)

    # ---- report: old rule vs new rule ---------------------------------------
    try:
        def _est_tokens(mask):
            return sum(len(docs[int(i)]) for i in np.where(mask)[0]) / cpt

        nb = (new_bad & other_ok) if used_new else np.zeros(n, dtype=bool)
        both = old_bad & nb
        print(
            "[curate] word-order test: threshold=%.4f upper_class_median=%.4f (%s)"
            % (new_thr, upper_med, "new rule active" if used_new else "fallback to old rule"),
            flush=True,
        )
        print(
            "[curate] dropped by old rule: %d docs, ~%d tokens | "
            "dropped by new rule: %d docs, ~%d tokens | both: %d docs"
            % (
                int(old_bad.sum()),
                int(_est_tokens(old_bad)),
                int(nb.sum()),
                int(_est_tokens(nb)),
                int(both.sum()),
            ),
            flush=True,
        )
    except Exception:
        pass

    budget = 1.02 * T  # small margin for the estimate; overshoot is truncated

    selected = []
    acc = 0.0
    for i in ranked:
        i = int(i)
        if acc >= budget:
            break
        if score[i] < 0.5 and acc >= 0.5 * T:
            # in-domain docs exhausted but enough to cycle: stop here
            break
        selected.append(i)
        acc += len(docs[i]) / cpt

    if not selected:
        selected = [int(i) for i in ranked]

    # ---- (6) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
