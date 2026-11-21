"""Curator: target-similarity filter with dedup, junk removal, a word-order
filter and a second-stage "clean target vs. surviving candidates" discriminator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop exact / near duplicates (hash of normalised first 500 chars, keep first).
  2. Drop junk (too short, low unique-word ratio, low alphabetic/space share).
  3. Word-order filter: drop docs whose adjacent word pairs rarely occur in the
     target sample (catches articles whose words are shuffled within sentences,
     which a char n-gram classifier cannot see).
       - words = lower-cased whitespace split of the first 3000 chars
       - B = adjacent word bigrams occurring >= 2 times across the full text of
         all target_sample docs
       - rate(doc) = fraction of the doc's adjacent word pairs that are in B
       - calibration: 10-fold cross-fit over target_sample (folds drawn from
         the seed + 7919 stream). Each held-out doc is scored on its first
         3000 chars against the bigram set (count >= 2, full text) built from
         the other 9 folds.
       - thr = 2nd percentile of the held-out target rates (no extra factor)
       - drop docs with rate < thr
  4. First stage: score remaining docs with a char n-gram TF-IDF + logistic
     regression (target_sample = positives, seeded random pool sample =
     negatives).
  5. Second stage (new): H = survivors with first-stage score >= 0.5.
       - positives = target_sample, negatives = seeded random sample of
         min(3000, |H|) docs from H; first 1500 chars of each
       - TfidfVectorizer(char_wb, (2,5), sublinear_tf, min_df=2,
         max_features=200000, float32) fitted on this training set
       - LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000,
         random_state=seed)
       - 5-fold cross-fit (folds from the seed + 104729 stream, drawn
         separately over positives and negatives so each fold holds 1/5 of
         each); thr2 = 5th percentile of the held-out target decision values
       - final model fitted on the whole training set scores all of H
         (training negatives included, scored like the rest)
       - docs in H with decision value < thr2 are flagged "off-target"
  6. Fill order: non-flagged H docs by first-stage score descending; then
     flagged H docs by second-stage value descending; then the remaining
     survivors (score < 0.5) by first-stage score descending. Take docs until
     the token budget T is filled.
  7. Shuffle the selection with the CURATE_SEED seed.

Deviations from the written spec, made to stay inside the 180 s limit or to
avoid discarding target-like text:
  - The first-stage vectorizer is fitted on the 3200 training docs (targets +
    negatives), not on the whole pool.
  - The prefix used to score pool docs shrinks (first stage: below 2000 chars,
    deadline 80 s; second stage: below 1500 chars, deadline 140 s) if time is
    short; the minimum is 200 chars.
  - Junk statistics are computed on the first 3000 chars of each doc.
  - The junk thresholds (0.2 unique-word ratio, 0.8 alphabetic/space share) are
    loosened if the target sample itself falls below them.
  - The word-order filter is skipped if it cannot be calibrated (fewer than 20
    usable held-out target rates) or if it would remove every surviving doc.
  - The second stage is skipped (fill order = first-stage score, as before) if
    |H| < 500, if the first-stage classifier failed, if more than 105 s have
    passed when it would start, if fewer than 20 held-out target values are
    available, if it raises, or if it runs so late that the 180 s limit is at
    risk (training not finished by 130 s, or scoring of H not finished by
    152 s). Every skip is logged with its reason.
  - The negatives of the second stage are sampled from their own stream
    (seed + 1299709) so the main stream and the fold stream are untouched.
  - The report-only scoring of docs removed by the word-order filter is gone
    (time is used by the second stage instead).

Diagnostics (printed to stderr, prefix "[curate]"):
  - word-order: old/new thresholds, held-out target percentiles, pool
    histogram, docs/tokens removed, docs/tokens newly removed vs old threshold
  - stage2: held-out target decision percentiles (5/50/95) and thr2;
    histogram of the second-stage values over H; docs and tokens (fraction of
    T) flagged; flag rate among training negatives (scored in-sample) vs the
    other H docs; shortest prefix used when scoring H; first 150 chars of the
    10 flagged docs with the lowest values
  - fill: docs and tokens of each tier that ended up in the selection
  - tokens (as a fraction of T) of remaining docs with score >= 0.5
  - total time
  Token figures in the report are exact for docs tokenised during the fill and
  estimated from the measured tokens-per-char ratio for the rest.
"""
import hashlib
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

PREFIX = 2000          # chars used for the first-stage classifier features
JUNK_WINDOW = 3000     # chars used for junk statistics
ORDER_WINDOW = 3000    # chars used for the word-order rate
ORDER_MIN_COUNT = 2    # min occurrences of a bigram in the target sample
ORDER_FOLDS = 10       # cross-fit folds for the threshold calibration
ORDER_PCT = 2.0        # percentile of held-out target rates
ORDER_FACTOR = 1.0     # thr = ORDER_FACTOR * percentile
OLD_ORDER_PCT = 5.0    # previous calibration (reported for comparison only)
OLD_ORDER_FACTOR = 0.7
FEATURE_DEADLINE = 80.0    # seconds since start by which stage-1 scoring ends

S2_PREFIX = 1500           # chars used for the second-stage features
S2_MIN_H = 500             # min |H| to run the second stage
S2_MAX_NEG = 3000          # max negatives sampled from H
S2_FOLDS = 5               # cross-fit folds
S2_PCT = 5.0               # percentile of held-out target decision values
S2_SCORE_THR = 0.5         # first-stage score defining H
S2_START_DEADLINE = 105.0  # skip the stage if it would start later than this
S2_TRAIN_ABORT = 130.0     # abandon the stage if training is not done by then
S2_SCORE_DEADLINE = 140.0  # scoring of H should finish by this time
S2_HARD_DEADLINE = 152.0   # abandon the stage if scoring is still running
S2_NEG_STREAM = 1299709    # seed offset for sampling the negatives
S2_FOLD_STREAM = 104729    # seed offset for the cross-fit folds

TOKEN_DEADLINE = 160.0     # seconds since start after which we stop tokenising

_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")

_TIER_NAMES = {
    0: "single tier (first-stage score order)",
    1: "tier 1: H, not flagged (by first-stage score)",
    2: "tier 2: H, flagged off-target (by second-stage value)",
    3: "tier 3: survivors with first-stage score < 0.5",
}


def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _stats(text):
    """Return (unique-word ratio, alphabetic/space char ratio) on a window."""
    s = text[:JUNK_WINDOW]
    n = len(s)
    if n == 0:
        return 0.0, 0.0
    words = s.lower().split()
    uniq = len(set(words)) / len(words) if words else 0.0
    non_alpha = len(_ALPHA_SPACE.sub("", s))
    return uniq, 1.0 - non_alpha / n


def _doc_bigrams(text):
    """Counter of adjacent word bigrams over the full text."""
    w = text.lower().split()
    if len(w) > 1:
        return Counter(zip(w, w[1:]))
    return Counter()


def _order_rate(text, bset, exclude=None):
    """Fraction of adjacent word pairs (first ORDER_WINDOW chars) in bset.

    Pairs in `exclude` are treated as not being in bset.
    Returns None if the window has fewer than two words.
    """
    w = text[:ORDER_WINDOW].lower().split()
    n = len(w) - 1
    if n <= 0:
        return None
    hit = 0
    if exclude:
        for b in zip(w, w[1:]):
            if b in bset and b not in exclude:
                hit += 1
    else:
        for b in zip(w, w[1:]):
            if b in bset:
                hit += 1
    return hit / n


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)
    if n_docs == 0:
        return []

    # Junk thresholds, relaxed if the target itself looks unusual
    uniq_thr, alpha_thr = 0.2, 0.8
    tstats = [_stats(t) for t in target_sample if len(t) >= 200]
    if len(tstats) >= 20:
        tu = np.array([s[0] for s in tstats])
        ta = np.array([s[1] for s in tstats])
        uniq_thr = min(uniq_thr, float(np.quantile(tu, 0.05)))
        alpha_thr = min(alpha_thr, float(np.quantile(ta, 0.05)))

    # Dedup and junk filter
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if len(d) < 200:
            continue
        key = hashlib.md5(
            " ".join(d[:500].lower().split()).encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        uniq, alpha = _stats(d)
        if uniq < uniq_thr or alpha < alpha_thr:
            continue
        survivors.append(i)
    del seen

    if not survivors:
        idx = list(range(n_docs))
        rng.shuffle(idx)
        return idx

    # Word-order (shuffled-text) filter
    removed = []            # removed doc indices, highest rate first
    removed_rates = None    # word-order rates aligned with `removed`
    order_thr = None        # new threshold (10-fold, 2nd percentile)
    old_thr = None          # previous threshold, for the report only
    try:
        tw0 = time.time()
        tgt = list(target_sample)
        perm = list(range(len(tgt)))
        random.Random(seed + 7919).shuffle(perm)   # own stream; rng untouched

        bg = [_doc_bigrams(t) for t in tgt]
        total_cnt = Counter()
        for c in bg:
            total_cnt.update(c)
        full_set = {b for b, c in total_cnt.items() if c >= ORDER_MIN_COUNT}

        # Previous calibration (two halves), reported for comparison only.
        try:
            half = len(perm) // 2
            idx_a, idx_b = perm[:half], perm[half:]
            held_old = []
            if idx_a and idx_b:
                cnt_a = Counter()
                for i in idx_a:
                    cnt_a.update(bg[i])
                set_a = {b for b, c in cnt_a.items() if c >= ORDER_MIN_COUNT}
                set_b = {b for b, c in total_cnt.items()
                         if c - cnt_a.get(b, 0) >= ORDER_MIN_COUNT}
                for i in idx_a:
                    r = _order_rate(tgt[i], set_b)
                    if r is not None:
                        held_old.append(r)
                for i in idx_b:
                    r = _order_rate(tgt[i], set_a)
                    if r is not None:
                        held_old.append(r)
                del cnt_a, set_a, set_b
            if len(held_old) >= 20:
                old_p5 = float(np.percentile(
                    np.array(held_old, dtype=np.float64), OLD_ORDER_PCT))
                old_thr = OLD_ORDER_FACTOR * old_p5
                _log("word-order: OLD calibration (two halves, n=%d) "
                     "p5=%.4f -> old thr = %.1f * p5 = %.4f"
                     % (len(held_old), old_p5, OLD_ORDER_FACTOR, old_thr))
            else:
                _log("word-order: OLD calibration not computable "
                     "(%d usable rates)" % len(held_old))
        except Exception as e:
            old_thr = None
            _log("word-order: OLD calibration failed (%r)" % (e,))

        # New calibration: k-fold cross-fit.
        n_folds = min(ORDER_FOLDS, len(perm))
        held = []
        if n_folds >= 2:
            for k in range(n_folds):
                fold = perm[k::n_folds]
                fold_cnt = Counter()
                for i in fold:
                    fold_cnt.update(bg[i])
                # bigrams that fall below the min count without this fold
                excl = {b for b, c in fold_cnt.items()
                        if total_cnt[b] - c < ORDER_MIN_COUNT}
                for i in fold:
                    r = _order_rate(tgt[i], full_set, excl)
                    if r is not None:
                        held.append(r)
        del bg, total_cnt

        if len(held) >= 20:
            held = np.array(held, dtype=np.float64)
            p2, p5, p50, p95 = (
                float(x) for x in np.percentile(held, [2, 5, 50, 95])
            )
            order_thr = ORDER_FACTOR * float(np.percentile(held, ORDER_PCT))
            _log(
                "word-order: target held-out rate (%d-fold, n=%d) p2=%.4f "
                "p5=%.4f p50=%.4f p95=%.4f"
                % (n_folds, len(held), p2, p5, p50, p95)
            )
            _log("word-order: NEW thr = %.4f (old thr = %s)"
                 % (order_thr,
                    "%.4f" % old_thr if old_thr is not None else "n/a"))
            if old_thr is not None and old_thr > order_thr:
                _log("word-order: NOTE new threshold is LOWER than the old "
                     "one; the filter is looser than before")

            rates = np.zeros(len(survivors), dtype=np.float64)
            for k, i in enumerate(survivors):
                r = _order_rate(docs[i], full_set)
                rates[k] = 0.0 if r is None else r
            del full_set

            # Pool histogram (docs that passed dedup + junk)
            hi = max(float(rates.max()), order_thr * 2.0, 1e-6)
            hist, edges = np.histogram(rates, bins=20, range=(0.0, hi))
            _log("word-order: pool rate histogram over %d docs "
                 "(after dedup+junk):" % len(survivors))
            for h, lo, up in zip(hist, edges[:-1], edges[1:]):
                mark = " <- thr" if lo <= order_thr < up else ""
                if old_thr is not None and lo <= old_thr < up:
                    mark += " <- old thr"
                _log("    [%.4f, %.4f) %7d%s" % (lo, up, int(h), mark))

            keep_mask = rates >= order_thr
            kept = [i for i, m in zip(survivors, keep_mask) if m]
            if kept:
                drop_idx = [i for i, m in zip(survivors, keep_mask) if not m]
                drop_rates = rates[~keep_mask]
                o = np.argsort(-drop_rates, kind="stable")
                new_removed = [drop_idx[j] for j in o.tolist()]
                new_removed_rates = drop_rates[o]
                survivors = kept
                removed = new_removed
                removed_rates = new_removed_rates
                _log("word-order: removed %d docs, kept %d (%.1fs)"
                     % (len(removed), len(survivors), time.time() - tw0))
            else:
                _log("word-order: filter would remove everything; skipped")
        else:
            _log("word-order: only %d usable held-out target rates; "
                 "filter skipped" % len(held))
    except Exception as e:  # never let the filter break curation
        removed = []
        removed_rates = None
        _log("word-order: filter failed (%r); skipped" % (e,))

    def _score_with(vec_, fn, idx_list, deadline, prefix, st,
                    hard_deadline=None):
        """Score docs[idx] with fn(vec_.transform(prefixes)).

        The prefix shrinks (down to 200 chars) to finish by `deadline`.
        If `hard_deadline` is given and passed, scoring stops and the
        unscored entries stay NaN.
        """
        out = np.full(len(idx_list), np.nan, dtype=np.float64)
        chunk = 2000
        for start in range(0, len(idx_list), chunk):
            now = time.time() - t0
            if hard_deadline is not None and now > hard_deadline:
                break
            part = idx_list[start:start + chunk]
            remaining_docs = len(idx_list) - start
            remaining_time = deadline - now
            if st["chars"] > 0:
                st["rate"] = st["time"] / st["chars"]
            if remaining_time <= 0:
                L = 200
            else:
                L = int(remaining_time
                        / (st["rate"] * remaining_docs + 1e-12))
                L = max(200, min(prefix, L))
            st["min_len"] = min(st.get("min_len", prefix), L)
            texts = [docs[i][:L] for i in part]
            tc = time.time()
            X = vec_.transform(texts)
            out[start:start + len(part)] = fn(X)
            st["time"] += time.time() - tc
            st["chars"] += sum(len(t) for t in texts)
        return out

    # First-stage classifier (target vs. random pool sample)
    scores = None
    try:
        neg_idx = rng.sample(range(n_docs), min(3000, n_docs))
        pos_txt = [t[:PREFIX] for t in target_sample]
        neg_txt = [docs[i][:PREFIX] for i in neg_idx]
        train_txt = pos_txt + neg_txt
        y = np.array([1] * len(pos_txt) + [0] * len(neg_txt))

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=50000,
            dtype=np.float32,
        )
        tf0 = time.time()
        Xtr = vec.fit_transform(train_txt)
        fit_time = time.time() - tf0
        train_chars = sum(len(t) for t in train_txt) or 1
        st1 = {
            "rate": fit_time / train_chars,  # seconds per char (pessimistic)
            "chars": 0,
            "time": 0.0,
        }

        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
        )
        clf.fit(Xtr, y)
        pos_col = list(clf.classes_).index(1)
        del Xtr, train_txt, pos_txt, neg_txt

        scores = _score_with(
            vec, lambda X: clf.predict_proba(X)[:, pos_col],
            survivors, FEATURE_DEADLINE, PREFIX, st1)
        _log("stage1: scored %d docs, shortest prefix %d chars, done at "
             "%.1fs" % (len(survivors), st1.get("min_len", PREFIX),
                        time.time() - t0))
        del vec, clf
    except Exception as e:
        scores = None
        _log("stage1: classifier failed (%r)" % (e,))

    # Second stage: clean target vs. surviving candidates
    s2 = None   # dict with the results if the stage ran to completion
    if scores is None:
        _log("stage2: skipped (first-stage classifier failed)")
    else:
        try:
            ts0 = time.time()
            h_pos = np.flatnonzero(scores >= S2_SCORE_THR)
            elapsed = ts0 - t0
            if len(h_pos) < S2_MIN_H:
                _log("stage2: skipped (|H| = %d < %d)"
                     % (len(h_pos), S2_MIN_H))
            elif elapsed > S2_START_DEADLINE:
                _log("stage2: skipped (%.1fs elapsed > %.0fs)"
                     % (elapsed, S2_START_DEADLINE))
            else:
                n_neg = min(S2_MAX_NEG, len(h_pos))
                neg_pos = random.Random(seed + S2_NEG_STREAM).sample(
                    h_pos.tolist(), n_neg)
                pos_txt2 = [t[:S2_PREFIX] for t in target_sample]
                neg_txt2 = [docs[survivors[p]][:S2_PREFIX] for p in neg_pos]
                n_pos = len(pos_txt2)
                train2 = pos_txt2 + neg_txt2
                y2 = np.array([1] * n_pos + [0] * n_neg)

                vec2 = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=200000,
                    dtype=np.float32,
                )
                tf0 = time.time()
                X2 = vec2.fit_transform(train2).tocsr()
                fit_time2 = time.time() - tf0
                train_chars2 = sum(len(t) for t in train2) or 1
                del train2, pos_txt2, neg_txt2
                _log("stage2: |H| = %d, training set %d pos + %d neg, "
                     "%d features, vectorizer fit %.1fs"
                     % (len(h_pos), n_pos, n_neg, X2.shape[1], fit_time2))

                # 5-fold cross-fit; folds drawn separately over positives
                # and negatives from the seed + 104729 stream
                rf = random.Random(seed + S2_FOLD_STREAM)
                pi = list(range(n_pos))
                rf.shuffle(pi)
                ni = list(range(n_neg))
                rf.shuffle(ni)
                fold_id = np.zeros(n_pos + n_neg, dtype=np.int64)
                for k, i in enumerate(pi):
                    fold_id[i] = k % S2_FOLDS
                for k, i in enumerate(ni):
                    fold_id[n_pos + i] = k % S2_FOLDS

                held2 = np.full(n_pos, np.nan, dtype=np.float64)
                aborted = False
                for k in range(S2_FOLDS):
                    if (time.time() - t0) > S2_TRAIN_ABORT:
                        aborted = True
                        break
                    te = fold_id == k
                    tr = np.flatnonzero(~te)
                    te_pos = np.flatnonzero(te[:n_pos])
                    if len(te_pos) == 0 or len(set(y2[tr].tolist())) < 2:
                        continue
                    m = LogisticRegression(
                        C=0.5, class_weight="balanced", max_iter=1000,
                        random_state=seed)
                    m.fit(X2[tr], y2[tr])
                    held2[te_pos] = m.decision_function(X2[te_pos])
                    del m

                held_ok = held2[~np.isnan(held2)]
                if aborted or (time.time() - t0) > S2_TRAIN_ABORT:
                    _log("stage2: ABANDONED (cross-fit not finished by "
                         "%.0fs; %.1fs elapsed); fill order = first-stage "
                         "score" % (S2_TRAIN_ABORT, time.time() - t0))
                elif len(held_ok) < 20:
                    _log("stage2: skipped (only %d held-out target values)"
                         % len(held_ok))
                else:
                    q5, q50, q95 = (float(x) for x in
                                    np.percentile(held_ok, [5, 50, 95]))
                    thr2 = float(np.percentile(held_ok, S2_PCT))
                    _log("stage2: held-out target decision values "
                         "(%d-fold, n=%d) p5=%.4f p50=%.4f p95=%.4f"
                         % (S2_FOLDS, len(held_ok), q5, q50, q95))
                    _log("stage2: thr2 = %.4f (percentile %.0f)"
                         % (thr2, S2_PCT))

                    clf2 = LogisticRegression(
                        C=0.5, class_weight="balanced", max_iter=1000,
                        random_state=seed)
                    clf2.fit(X2, y2)
                    del X2
                    _log("stage2: training done at %.1fs"
                         % (time.time() - t0))

                    st2 = {
                        "rate": fit_time2 / train_chars2,
                        "chars": 0,
                        "time": 0.0,
                    }
                    h_idx = [survivors[p] for p in h_pos.tolist()]
                    dvals = _score_with(
                        vec2, clf2.decision_function, h_idx,
                        S2_SCORE_DEADLINE, S2_PREFIX, st2,
                        hard_deadline=S2_HARD_DEADLINE)
                    n_nan = int(np.isnan(dvals).sum())
                    if n_nan > 0:
                        _log("stage2: ABANDONED (%d of %d H docs not scored "
                             "by %.0fs); fill order = first-stage score"
                             % (n_nan, len(h_idx), S2_HARD_DEADLINE))
                    else:
                        is_train_neg = np.zeros(len(survivors), dtype=bool)
                        is_train_neg[np.array(neg_pos, dtype=np.int64)] = True
                        s2 = {
                            "h_pos": h_pos,
                            "dvals": dvals,
                            "thr2": thr2,
                            "flagged": dvals < thr2,
                            "train_neg": is_train_neg[h_pos],
                            "min_len": st2.get("min_len", S2_PREFIX),
                            "pct": (q5, q50, q95),
                        }
                        _log("stage2: scored %d H docs, shortest prefix %d "
                             "chars, done at %.1fs (stage took %.1fs)"
                             % (len(h_idx), s2["min_len"],
                                time.time() - t0, time.time() - ts0))
                    del vec2, clf2
        except Exception as e:
            s2 = None
            _log("stage2: failed (%r); fill order = first-stage score" % (e,))

    # Fill order (positions into `survivors`) and the tier of each position
    tier = np.zeros(len(survivors), dtype=np.int64)
    if scores is None:
        # Classifier failed: keep the filtered pool in seeded random order
        order = list(range(len(survivors)))
        rng.shuffle(order)
    elif s2 is None:
        order = np.argsort(-scores, kind="stable").tolist()
    else:
        h_pos = s2["h_pos"]
        flagged = s2["flagged"]
        dvals = s2["dvals"]
        clean_pos = h_pos[~flagged]
        flag_pos = h_pos[flagged]
        in_h = np.zeros(len(survivors), dtype=bool)
        in_h[h_pos] = True
        rest_pos = np.flatnonzero(~in_h)
        o1 = clean_pos[np.argsort(-scores[clean_pos], kind="stable")]
        o2 = flag_pos[np.argsort(-dvals[flagged], kind="stable")]
        o3 = rest_pos[np.argsort(-scores[rest_pos], kind="stable")]
        tier[o1] = 1
        tier[o2] = 2
        tier[o3] = 3
        order = o1.tolist() + o2.tolist() + o3.tolist()

    # Fill the token budget from the top
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0
    batch = 512
    pos = 0
    n_surv = len(order)
    exact_counts = {}
    exact_tok, exact_chars = 0, 0
    sel_tier_tok = Counter()
    sel_tier_docs = Counter()
    while pos < n_surv and total < T:
        part_pos = order[pos:pos + batch]
        part = [survivors[j] for j in part_pos]
        pos += batch
        counts = None
        exact = False
        if enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in part], num_threads=8
                )
                counts = [len(t) for t in toks]
                exact = True
            except Exception:
                counts = None
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in part]
        if exact:
            for i, c in zip(part, counts):
                exact_counts[i] = c
                exact_tok += c
                exact_chars += len(docs[i])
        for j, i, c in zip(part_pos, part, counts):
            selected.append(i)
            total += c
            tj = int(tier[j])
            sel_tier_tok[tj] += c
            sel_tier_docs[tj] += 1
            if total >= T:
                break

    if not selected:
        selected = list(survivors)

    # Diagnostics
    try:
        tpc = (exact_tok / exact_chars) if exact_chars > 0 else 0.25

        def _tok(i):
            c = exact_counts.get(i)
            return c if c is not None else len(docs[i]) * tpc

        _log("fill: selected %d docs, %d tokens (%.3f T); tokens/char=%.4f "
             "(measured on %d docs)"
             % (len(selected), total, total / float(T), tpc,
                len(exact_counts)))
        if total < T:
            _log("WARNING: the whole filtered pool holds only %.3f T tokens; "
                 "the trainer will cycle the selection" % (total / float(T)))
        for tj in sorted(sel_tier_docs):
            _log("fill: %s -> %d docs, %d tokens (%.3f T) in the selection"
                 % (_TIER_NAMES.get(tj, "tier %d" % tj), sel_tier_docs[tj],
                    sel_tier_tok[tj], sel_tier_tok[tj] / float(T)))

        if order_thr is not None:
            _log("word-order: thresholds old=%s new=%.4f"
                 % ("%.4f" % old_thr if old_thr is not None else "n/a",
                    order_thr))
            rem_tok = sum(len(docs[i]) * tpc for i in removed)
            _log("word-order: removed in total %d docs, ~%d tokens "
                 "(est., %.3f T)"
                 % (len(removed), int(rem_tok), rem_tok / float(T)))
            if old_thr is None:
                _log("word-order: newly removed docs (vs old threshold): "
                     "n/a, old threshold not computable")
            elif removed_rates is None or not removed:
                _log("word-order: newly removed docs (old threshold would "
                     "have kept): 0 docs, 0 tokens (0.000 T)")
            else:
                new_mask = removed_rates >= old_thr
                new_idx = [i for i, m in zip(removed, new_mask) if m]
                new_tok = sum(len(docs[i]) * tpc for i in new_idx)
                _log("word-order: newly removed docs (old threshold would "
                     "have kept, %.4f <= rate < %.4f): %d docs, ~%d tokens "
                     "(est., %.3f T)"
                     % (old_thr, order_thr, len(new_idx), int(new_tok),
                        new_tok / float(T)))

        if scores is not None:
            hi_idx = [i for i, s in zip(survivors, scores) if s >= 0.5]
            n_exact = sum(1 for i in hi_idx if i in exact_counts)
            hi_tok = sum(_tok(i) for i in hi_idx)
            _log("remaining docs with score >= 0.5: %d docs, ~%d tokens "
                 "= %.3f T (%d docs counted exactly, rest estimated)"
                 % (len(hi_idx), int(hi_tok), hi_tok / float(T), n_exact))
            if hi_tok < T:
                _log("WARNING: remaining docs with classifier score >= 0.5 "
                     "hold FEWER than T tokens (%.3f T, short by ~%d "
                     "tokens); the budget is filled with the next best "
                     "scored docs (score < 0.5)"
                     % (hi_tok / float(T), int(T - hi_tok)))
        else:
            _log("WARNING: classifier unavailable; cannot check whether "
                 "docs with score >= 0.5 fill the budget")

        if s2 is None:
            _log("stage2: NOT APPLIED; selection uses first-stage order only")
        else:
            h_pos = s2["h_pos"]
            dvals = s2["dvals"]
            flagged = s2["flagged"]
            thr2 = s2["thr2"]
            q5, q50, q95 = s2["pct"]
            _log("stage2: held-out target decision p5=%.4f p50=%.4f "
                 "p95=%.4f; thr2=%.4f" % (q5, q50, q95, thr2))
            if s2["min_len"] < S2_PREFIX:
                _log("stage2: NOTE prefix was shortened to as little as %d "
                     "chars (of %d) while scoring H; values of those docs "
                     "are less comparable with thr2"
                     % (s2["min_len"], S2_PREFIX))

            lo_v, hi_v = float(dvals.min()), float(dvals.max())
            if hi_v <= lo_v:
                hi_v = lo_v + 1e-6
            hist, edges = np.histogram(dvals, bins=20, range=(lo_v, hi_v))
            _log("stage2: histogram of second-stage values over H "
                 "(%d docs):" % len(dvals))
            for h, lo, up in zip(hist, edges[:-1], edges[1:]):
                mark = " <- thr2" if lo <= thr2 < up else ""
                _log("    [%9.4f, %9.4f) %7d%s" % (lo, up, int(h), mark))
            if thr2 < lo_v:
                _log("    (thr2 is below the lowest value)")
            elif thr2 >= hi_v:
                _log("    (thr2 is above the highest value)")

            h_idx = [survivors[p] for p in h_pos.tolist()]
            h_tok = sum(_tok(i) for i in h_idx)
            fl_idx = [i for i, f in zip(h_idx, flagged) if f]
            fl_tok = sum(_tok(i) for i in fl_idx)
            n_fl = len(fl_idx)
            _log("stage2: flagged off-target %d of %d H docs (%.1f%%), "
                 "~%d tokens = %.3f T (H holds ~%.3f T; non-flagged H "
                 "holds ~%.3f T)"
                 % (n_fl, len(h_idx), 100.0 * n_fl / max(1, len(h_idx)),
                    int(fl_tok), fl_tok / float(T), h_tok / float(T),
                    (h_tok - fl_tok) / float(T)))
            if (h_tok - fl_tok) < T:
                _log("WARNING: non-flagged H docs hold FEWER than T tokens; "
                     "flagged docs (and possibly score < 0.5 docs) were "
                     "used to fill the budget")

            tn = s2["train_neg"]
            n_tn, n_other = int(tn.sum()), int((~tn).sum())
            f_tn = int((flagged & tn).sum())
            f_other = int((flagged & ~tn).sum())
            _log("stage2: flag rate among training negatives (scored "
                 "in-sample): %d of %d (%.1f%%); among other H docs: "
                 "%d of %d (%.1f%%)"
                 % (f_tn, n_tn, 100.0 * f_tn / max(1, n_tn),
                    f_other, n_other, 100.0 * f_other / max(1, n_other)))

            if n_fl > 0:
                low = np.argsort(dvals, kind="stable")[:10]
                _log("stage2: 10 flagged docs with the lowest values "
                     "(value, first-stage score, doc index, first 150 "
                     "chars):")
                for q in low.tolist():
                    if not flagged[q]:
                        continue
                    di = h_idx[q]
                    _log("    %9.4f  s1=%.3f  #%d  %r"
                         % (float(dvals[q]), float(scores[h_pos[q]]), di,
                            docs[di][:150]))
            else:
                _log("stage2: no docs flagged")

        _log("total time %.1fs" % (time.time() - t0))
    except Exception as e:
        _log("report failed: %r" % (e,))

    # Seeded shuffle
    random.Random(seed).shuffle(selected)
    return selected
