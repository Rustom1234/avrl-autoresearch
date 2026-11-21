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
       - thr = 0.7 * 5th percentile of cross-fitted target rates (sample split
         in two halves, each half scored against the other half's bigram set)
       - drop docs with rate < thr
  4. Score remaining docs with a char n-gram TF-IDF + logistic regression
     (target_sample = positives, seeded random pool sample = negatives).
  5. Take the best scored docs until the token budget T is filled.
  6. Order the selection as two blocks by classifier score:
       - H = selected docs with score >= 0.5 (target-like)
       - L = selected docs with score <  0.5 (filler)
     Each block is shuffled separately with one CURATE_SEED-seeded RNG (L first,
     then H) and the result is L followed by H, so the filler is read first and
     the target-like text is read in the final steps of training. No doc is
     repeated and the set of selected docs is unchanged.
     Fallbacks to the previous single seeded shuffle of the whole selection:
       - L is empty or H is empty
       - the classifier failed (scores is None)
       - the split itself raises

Deviations from the written spec, made to stay inside the 180 s limit or to
avoid discarding target-like text:
  - The vectorizer is fitted on the 3200 training docs (targets + negatives),
    not on the whole pool.
  - The prefix used to score pool docs shrinks below 2000 chars if time is short.
  - Junk statistics are computed on the first 3000 chars of each doc, so long
    documents are not penalised for a naturally lower unique-word ratio.
  - The junk thresholds (0.2 unique-word ratio, 0.8 alphabetic/space share) are
    loosened if the target sample itself falls below them.
  - The word-order filter is skipped if it cannot be calibrated (too few usable
    target docs) or if it would remove every surviving doc.

Diagnostics (printed to stderr, prefix "[curate]"):
  - target held-out rate percentiles (5/50/95) and the threshold
  - histogram of the word-order rate over the pool docs that passed dedup+junk
  - docs and tokens removed by the word-order filter
  - how many removed docs had classifier score >= 0.5 (removed docs are scored
    for the report only, after the kept docs, and only while time allows)
  - tokens (as a fraction of T) of remaining docs with score >= 0.5
  - docs and tokens (as a fraction of T) in block H and in block L of the
    selection, and which ordering was used (L-then-H or single shuffle)
  - a warning if the selection holds fewer than T tokens: the trainer then
    cycles the list, so training would end on the start of the list (block L)
    rather than on block H
  Token figures in the report are exact for docs tokenised during the fill and
  estimated from the measured tokens-per-char ratio for the rest. The last
  selected doc may overshoot T; the trainer truncates at T.
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
ORDER_PCT = 5.0        # percentile of held-out target rates
ORDER_FACTOR = 0.7     # thr = ORDER_FACTOR * percentile
BLOCK_SCORE = 0.5      # classifier score splitting H (>=) from L (<)
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


def _bigram_set(texts, min_count=ORDER_MIN_COUNT):
    """Adjacent word bigrams occurring >= min_count times across full texts."""
    cnt = Counter()
    for t in texts:
        w = t.lower().split()
        if len(w) > 1:
            cnt.update(zip(w, w[1:]))
    return {b for b, c in cnt.items() if c >= min_count}


def _order_rate(text, bset):
    """Fraction of adjacent word pairs (first ORDER_WINDOW chars) in bset.

    Returns None if the window has fewer than two words.
    """
    w = text[:ORDER_WINDOW].lower().split()
    n = len(w) - 1
    if n <= 0:
        return None
    hit = 0
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
    removed = []
    order_thr = None
    try:
        tw0 = time.time()
        tgt = list(target_sample)
        perm = list(range(len(tgt)))
        random.Random(seed + 7919).shuffle(perm)   # own stream; rng untouched
        half = len(perm) // 2
        idx_a, idx_b = perm[:half], perm[half:]
        held = []
        if idx_a and idx_b:
            set_a = _bigram_set([tgt[i] for i in idx_a])
            set_b = _bigram_set([tgt[i] for i in idx_b])
            for i in idx_a:
                r = _order_rate(tgt[i], set_b)
                if r is not None:
                    held.append(r)
            for i in idx_b:
                r = _order_rate(tgt[i], set_a)
                if r is not None:
                    held.append(r)
            del set_a, set_b
        if len(held) >= 20:
            held = np.array(held, dtype=np.float64)
            p5, p50, p95 = (float(x) for x in np.percentile(held, [5, 50, 95]))
            order_thr = ORDER_FACTOR * float(np.percentile(held, ORDER_PCT))
            _log(
                "word-order: target held-out rate (n=%d) p5=%.4f p50=%.4f "
                "p95=%.4f -> thr=%.4f" % (len(held), p5, p50, p95, order_thr)
            )
            full_set = _bigram_set(tgt)
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
                _log("    [%.4f, %.4f) %7d%s" % (lo, up, int(h), mark))

            keep_mask = rates >= order_thr
            kept = [i for i, m in zip(survivors, keep_mask) if m]
            dropped = [i for i, m in zip(survivors, keep_mask) if not m]
            if kept:
                survivors, removed = kept, dropped
                _log("word-order: removed %d docs, kept %d (%.1fs)"
                     % (len(removed), len(survivors), time.time() - tw0))
            else:
                _log("word-order: filter would remove everything; skipped")
        else:
            _log("word-order: only %d usable held-out target rates; "
                 "filter skipped" % len(held))
    except Exception as e:  # never let the filter break curation
        removed = []
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
        if order_thr is not None:
            rem_tok = sum(len(docs[i]) * tpc for i in removed)
            _log("word-order: removed %d docs, ~%d tokens (est., %.3f T)"
                 % (len(removed), int(rem_tok), rem_tok / float(T)))
            if removed:
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
        if scores is not None:
            hi_idx = [i for i, s in zip(survivors, scores) if s >= 0.5]
            n_exact = sum(1 for i in hi_idx if i in exact_counts)
            hi_tok = sum(_tok(i) for i in hi_idx)
            _log("remaining docs with score >= 0.5: %d docs, ~%d tokens "
                 "= %.3f T (%d docs counted exactly, rest estimated)"
                 % (len(hi_idx), int(hi_tok), hi_tok / float(T), n_exact))
        _log("total time %.1fs" % (time.time() - t0))
    except Exception as e:
        _log("report failed: %r" % (e,))

    # Final ordering: filler block (L) first, target-like block (H) last.
    block_h, block_l = None, None
    if scores is not None:
        try:
            score_of = {i: float(s) for i, s in zip(survivors, scores)}
            block_h, block_l = [], []
            for i in selected:
                s = score_of.get(i)
                # NaN or missing scores count as filler
                if s is not None and s >= BLOCK_SCORE:
                    block_h.append(i)
                else:
                    block_l.append(i)
        except Exception as e:
            block_h, block_l = None, None
            _log("order: H/L split failed (%r); single shuffle" % (e,))

        if block_h is not None:
            try:
                tpc2 = (exact_tok / exact_chars) if exact_chars > 0 else 0.25

                def _tok2(i):
                    c = exact_counts.get(i)
                    return c if c is not None else len(docs[i]) * tpc2

                h_tok = sum(_tok2(i) for i in block_h)
                l_tok = sum(_tok2(i) for i in block_l)
                h_exact = sum(1 for i in block_h if i in exact_counts)
                l_exact = sum(1 for i in block_l if i in exact_counts)
                _log("order: block H (score >= %.2f): %d docs, ~%d tokens "
                     "= %.3f T (%d docs counted exactly)"
                     % (BLOCK_SCORE, len(block_h), int(h_tok),
                        h_tok / float(T), h_exact))
                _log("order: block L (score < %.2f, filler): %d docs, "
                     "~%d tokens = %.3f T (%d docs counted exactly)"
                     % (BLOCK_SCORE, len(block_l), int(l_tok),
                        l_tok / float(T), l_exact))
                if (h_tok + l_tok) < T:
                    _log("order: WARNING selection holds only %.3f T; the "
                         "trainer will cycle the list, so the last steps "
                         "re-read the start of the list (block L), not "
                         "block H" % ((h_tok + l_tok) / float(T)))
            except Exception as e:
                _log("order: H/L report failed: %r" % (e,))
    else:
        _log("order: classifier unavailable; H/L blocks not defined, "
             "single seeded shuffle")

    if block_h and block_l:
        brng = random.Random(seed)
        brng.shuffle(block_l)
        brng.shuffle(block_h)
        _log("order: returning L (%d docs) followed by H (%d docs)"
             % (len(block_l), len(block_h)))
        return block_l + block_h

    if block_h is not None:
        _log("order: %s block is empty; single seeded shuffle"
             % ("L" if not block_l else "H"))

    # Fallback: single seeded shuffle (previous behaviour)
    random.Random(seed).shuffle(selected)
    return selected
