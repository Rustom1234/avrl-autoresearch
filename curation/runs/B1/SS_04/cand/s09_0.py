"""Curator: target-similarity filter with dedup, junk removal and a word-order filter.

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
         the other 9 folds, which matches the size of the set pool docs are
         scored against.
       - thr = 2nd percentile of the held-out target rates (no extra factor)
       - drop docs with rate < thr
  4. Score remaining docs with a char n-gram TF-IDF + logistic regression
     (target_sample = positives, seeded random pool sample = negatives).
  5. Take the best scored docs until the token budget T is filled.
  6. Shuffle the selection with the CURATE_SEED seed.

Deviations from the written spec, made to stay inside the 180 s limit or to
avoid discarding target-like text:
  - The vectorizer is fitted on the 3200 training docs (targets + negatives),
    not on the whole pool.
  - The prefix used to score pool docs shrinks below 2000 chars if time is short.
  - Junk statistics are computed on the first 3000 chars of each doc, so long
    documents are not penalised for a naturally lower unique-word ratio.
  - The junk thresholds (0.2 unique-word ratio, 0.8 alphabetic/space share) are
    loosened if the target sample itself falls below them.
  - The word-order filter is skipped if it cannot be calibrated (fewer than 20
    usable held-out target rates) or if it would remove every surviving doc.
  - If the target sample has fewer than 10 docs, the number of folds is reduced
    to the number of docs (the filter is then skipped anyway: < 20 rates).

Diagnostics (printed to stderr, prefix "[curate]"):
  - old threshold (0.7 * 5th percentile of the two-half cross-fit, same
    permutation as before, computed for comparison only) and new threshold
  - target held-out rate percentiles (2/5/50/95) of the 10-fold cross-fit
  - histogram of the word-order rate over the pool docs that passed dedup+junk
  - docs and estimated tokens removed by the word-order filter in total
  - docs and estimated tokens removed that the old threshold would have kept,
    and how many of those had classifier score >= 0.5
  - how many removed docs had classifier score >= 0.5 (removed docs are scored
    for the report only, after the kept docs, highest word-order rate first so
    the newly removed docs are scored first, and only while time allows)
  - tokens (as a fraction of T) of remaining docs with score >= 0.5, with a
    WARNING line if they hold fewer than T tokens (the fill then continues
    with the next best scored docs, as before)
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

PREFIX = 2000          # chars used for the classifier features
JUNK_WINDOW = 3000     # chars used for junk statistics
ORDER_WINDOW = 3000    # chars used for the word-order rate
ORDER_MIN_COUNT = 2    # min occurrences of a bigram in the target sample
ORDER_FOLDS = 10       # cross-fit folds for the threshold calibration
ORDER_PCT = 2.0        # percentile of held-out target rates
ORDER_FACTOR = 1.0     # thr = ORDER_FACTOR * percentile
OLD_ORDER_PCT = 5.0    # previous calibration (reported for comparison only)
OLD_ORDER_FACTOR = 0.7
FEATURE_DEADLINE = 100.0   # seconds since start by which scoring should finish
REPORT_DEADLINE = 115.0    # scoring of removed docs (report only) stops here
TOKEN_DEADLINE = 160.0     # seconds since start after which we stop tokenising

_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")


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
                # highest rate first: docs the old threshold would have kept
                # are scored first for the report
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

    # Classifier
    scores = None
    removed_scores = None
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
        st = {
            "rate": fit_time / train_chars,  # seconds per char (pessimistic)
            "chars": 0,
            "time": 0.0,
        }

        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
        )
        clf.fit(Xtr, y)
        pos_col = list(clf.classes_).index(1)

        def _score(idx_list, deadline, stop_when_late):
            out = np.full(len(idx_list), np.nan, dtype=np.float64)
            chunk = 2000
            for start in range(0, len(idx_list), chunk):
                part = idx_list[start:start + chunk]
                remaining_docs = len(idx_list) - start
                remaining_time = deadline - (time.time() - t0)
                if st["chars"] > 0:
                    st["rate"] = st["time"] / st["chars"]
                if remaining_time <= 0:
                    if stop_when_late:
                        break
                    L = 200
                else:
                    L = int(remaining_time
                            / (st["rate"] * remaining_docs + 1e-12))
                    L = max(200, min(PREFIX, L))
                texts = [docs[i][:L] for i in part]
                tc = time.time()
                X = vec.transform(texts)
                out[start:start + len(part)] = clf.predict_proba(X)[:, pos_col]
                st["time"] += time.time() - tc
                st["chars"] += sum(len(t) for t in texts)
            return out

        scores = _score(survivors, FEATURE_DEADLINE, False)

        # Report only: score the removed docs if there is time left.
        if removed:
            try:
                removed_scores = _score(removed, REPORT_DEADLINE, True)
            except Exception:
                removed_scores = None
    except Exception:
        scores = None

    if scores is None:
        # Classifier failed: keep the filtered pool in seeded random order
        order = list(range(len(survivors)))
        rng.shuffle(order)
    else:
        order = np.argsort(-scores, kind="stable").tolist()

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
    while pos < n_surv and total < T:
        part = [survivors[j] for j in order[pos:pos + batch]]
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
        for i, c in zip(part, counts):
            selected.append(i)
            total += c
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

        if order_thr is not None:
            _log("word-order: thresholds old=%s new=%.4f"
                 % ("%.4f" % old_thr if old_thr is not None else "n/a",
                    order_thr))
            rem_tok = sum(len(docs[i]) * tpc for i in removed)
            _log("word-order: removed in total %d docs, ~%d tokens "
                 "(est., %.3f T)"
                 % (len(removed), int(rem_tok), rem_tok / float(T)))

            if removed:
                # All removed docs vs classifier score
                if removed_scores is not None:
                    done = ~np.isnan(removed_scores)
                    n_done = int(done.sum())
                    n_hi = int((removed_scores[done] >= 0.5).sum())
                    hi_tok = sum(
                        len(docs[i]) * tpc
                        for i, s in zip(removed, removed_scores)
                        if s == s and s >= 0.5
                    )
                    _log("word-order: removed docs with classifier score "
                         ">= 0.5: %d of %d scored (%d removed in total), "
                         "~%d tokens (est., %.3f T)"
                         % (n_hi, n_done, len(removed), int(hi_tok),
                            hi_tok / float(T)))
                else:
                    _log("word-order: removed docs were not scored "
                         "(classifier unavailable)")

            # Docs removed now that the old threshold would have kept
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
                if removed_scores is not None:
                    ns = removed_scores[new_mask]
                    ns_done = ~np.isnan(ns)
                    n_new_done = int(ns_done.sum())
                    n_new_hi = int((ns[ns_done] >= 0.5).sum())
                    new_hi_tok = sum(
                        len(docs[i]) * tpc
                        for i, s in zip(new_idx, ns)
                        if s == s and s >= 0.5
                    )
                    _log("word-order: newly removed docs with classifier "
                         "score >= 0.5: %d of %d scored (%d newly removed), "
                         "~%d tokens (est., %.3f T)"
                         % (n_new_hi, n_new_done, len(new_idx),
                            int(new_hi_tok), new_hi_tok / float(T)))
                    if n_new_done < len(new_idx):
                        _log("word-order: NOTE %d newly removed docs were "
                             "not scored (report deadline reached)"
                             % (len(new_idx) - n_new_done))
                else:
                    _log("word-order: newly removed docs were not scored "
                         "(classifier unavailable)")

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
        _log("total time %.1fs" % (time.time() - t0))
    except Exception as e:
        _log("report failed: %r" % (e,))

    # Seeded shuffle
    random.Random(seed).shuffle(selected)
    return selected
