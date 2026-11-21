"""Curator: target-similarity filter with dedup, junk removal, a word-order
filter and a pool-consensus fluency ranking of the fill.

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
  5. Pool-consensus fluency tiering of the fill order:
       - H = survivors with classifier score >= 0.5
       - bigram document frequency over H + target_sample (lower-cased
         whitespace split of the first 6000 chars, each adjacent word bigram
         counted once per doc); B2 = bigrams with document frequency >= 3
       - flu(doc) = fraction of the doc's adjacent word pairs (first 6000
         chars) that are in B2
       - thr = 10th percentile of flu over the target_sample docs (same B2)
       - Tier A: H with flu >= thr, by classifier score descending
         Tier B: H with flu <  thr, by flu descending
         Tier C: all other survivors, by classifier score descending
     Nothing is dropped; lower tiers are used only if the budget needs them.
  6. Take docs in that order until the token budget T is filled.
  7. Shuffle the selection with the CURATE_SEED seed.

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
  - The fluency tiering runs BEFORE the report-only scoring of the docs removed
    by the word-order filter (previously that scoring came straight after the
    survivor scoring). The selection is unaffected; the report-only scoring
    just has less time before its 115 s deadline, so that the tiering is not
    starved by a diagnostic.
  - The fluency tiering is skipped (the fill then uses plain classifier-score
    order, as before) if it passes 125 s since the start, if H or B2 is empty,
    if fewer than 20 target docs have a usable flu, or if it raises. Each case
    is logged as "fluency: ... SKIPPED".
  - Target docs with fewer than two words in the window are left out of the
    calibration; H docs with fewer than two words get flu = 0.

Diagnostics (printed to stderr, prefix "[curate]"):
  - target held-out rate percentiles (5/50/95) and the threshold
  - histogram of the word-order rate over the pool docs that passed dedup+junk
  - docs and tokens removed by the word-order filter
  - how many removed docs had classifier score >= 0.5 (removed docs are scored
    for the report only, after the kept docs, and only while time allows)
  - fluency: target flu percentiles (10/50/90) and thr, histogram of flu over
    H, docs and tokens (fraction of T) in tiers A/B/C, and the docs and tokens
    of each tier that ended up in the selection
  - tokens (as a fraction of T) of remaining docs with score >= 0.5
  Token figures in the report are exact for docs tokenised during the fill and
  estimated from the measured tokens-per-char ratio for the rest. The last
  selected doc is counted in full although training stops at T.
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
FLU_WINDOW = 6000      # chars used for the fluency score
FLU_MIN_DF = 3         # min document frequency of a bigram to be in B2
FLU_PCT = 10.0         # percentile of target flu used as the tier threshold
FLU_SCORE = 0.5        # classifier score defining H
FEATURE_DEADLINE = 100.0   # seconds since start by which scoring should finish
REPORT_DEADLINE = 115.0    # scoring of removed docs (report only) stops here
TIER_DEADLINE = 125.0      # fluency tiering is abandoned after this
TOKEN_DEADLINE = 160.0     # seconds since start after which we stop tokenising

_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")
_CODE_MULT = np.int64(1) << np.int64(31)


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


def _fluency_tiers(docs, target_sample, survivors, scores, t0):
    """Pool-consensus fluency tiering.

    Returns None if skipped, else a dict with
      order : list of positions into survivors (tier A, then B, then C)
      tier  : int8 array aligned with survivors (0 = A, 1 = B, 2 = C)
      thr   : flu threshold
    """
    def _late():
        return (time.time() - t0) > TIER_DEADLINE

    tf0 = time.time()
    scores = np.asarray(scores, dtype=np.float64)
    safe_scores = np.where(np.isnan(scores), -1.0, scores)
    h_pos = np.flatnonzero(safe_scores >= FLU_SCORE)
    n_h = int(h_pos.size)
    if n_h == 0:
        _log("fluency: H is empty; tiering SKIPPED")
        return None
    if _late():
        _log("fluency: no time left before step 2; tiering SKIPPED")
        return None

    wid = {}
    empty = np.zeros(0, dtype=np.int64)

    def _codes(text):
        w = text[:FLU_WINDOW].lower().split()
        if len(w) < 2:
            return empty
        ids = np.fromiter(
            (wid.setdefault(x, len(wid)) for x in w),
            dtype=np.int64, count=len(w),
        )
        return ids[:-1] * _CODE_MULT + ids[1:]

    # Step 2: per-doc bigram codes and document frequencies (H, then targets)
    n_t = len(target_sample)
    per_doc = []
    uniq = []
    for k in range(n_h + n_t):
        if (k & 255) == 0 and _late():
            _log("fluency: time guard hit in step 2 after %d of %d docs; "
                 "tiering SKIPPED" % (k, n_h + n_t))
            return None
        if k < n_h:
            text = docs[survivors[int(h_pos[k])]]
        else:
            text = target_sample[k - n_h]
        c = _codes(text)
        per_doc.append(c)
        if c.size:
            uniq.append(np.unique(c))
    del wid
    if not uniq:
        _log("fluency: no bigrams found; tiering SKIPPED")
        return None
    all_u = np.concatenate(uniq)
    del uniq
    vals, cnt = np.unique(all_u, return_counts=True)
    del all_u
    n_distinct = int(vals.size)
    b2 = vals[cnt >= FLU_MIN_DF]
    del vals, cnt
    if b2.size == 0:
        _log("fluency: B2 is empty; tiering SKIPPED")
        return None
    if _late():
        _log("fluency: time guard hit at end of step 2; tiering SKIPPED")
        return None

    # Step 3: flu for every doc (H and targets) against B2
    lens = np.array([c.size for c in per_doc], dtype=np.int64)
    allc = np.concatenate(per_doc) if int(lens.sum()) > 0 else empty
    del per_doc
    pos = np.searchsorted(b2, allc)
    pos[pos >= b2.size] = 0
    mask = b2[pos] == allc
    del pos, allc
    cs = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    del mask
    ends = np.cumsum(lens)
    starts = ends - lens
    hits = cs[ends] - cs[starts]
    flu_all = np.where(lens > 0, hits / np.maximum(lens, 1), 0.0)
    if _late():
        _log("fluency: time guard hit in step 3; tiering SKIPPED")
        return None

    h_flu = flu_all[:n_h]
    t_flu = flu_all[n_h:][lens[n_h:] > 0]

    # Step 4: calibrate on the target
    if t_flu.size < 20:
        _log("fluency: only %d usable target flu values; tiering SKIPPED"
             % int(t_flu.size))
        return None
    p10, p50, p90 = (float(x) for x in np.percentile(t_flu, [10, 50, 90]))
    thr = float(np.percentile(t_flu, FLU_PCT))
    _log("fluency: |H|=%d docs, %d target docs, %d distinct bigrams, "
         "|B2|=%d (df >= %d)"
         % (n_h, n_t, n_distinct, int(b2.size), FLU_MIN_DF))
    _log("fluency: target flu (n=%d) p10=%.4f p50=%.4f p90=%.4f -> thr=%.4f"
         % (int(t_flu.size), p10, p50, p90, thr))
    hist, edges = np.histogram(h_flu, bins=20, range=(0.0, 1.0))
    _log("fluency: flu histogram over H (%d docs):" % n_h)
    for h, lo, up in zip(hist, edges[:-1], edges[1:]):
        mark = " <- thr" if lo <= thr < up else ""
        _log("    [%.2f, %.2f) %7d%s" % (lo, up, int(h), mark))

    # Step 5: tiers
    in_a = h_flu >= thr
    a_pos = h_pos[in_a]
    b_pos = h_pos[~in_a]
    a_pos = a_pos[np.argsort(-safe_scores[a_pos], kind="stable")]
    b_pos = b_pos[np.argsort(-h_flu[~in_a], kind="stable")]
    is_h = np.zeros(len(survivors), dtype=bool)
    is_h[h_pos] = True
    c_pos = np.flatnonzero(~is_h)
    c_pos = c_pos[np.argsort(-safe_scores[c_pos], kind="stable")]

    tier = np.full(len(survivors), 2, dtype=np.int8)
    tier[a_pos] = 0
    tier[b_pos] = 1
    order = a_pos.tolist() + b_pos.tolist() + c_pos.tolist()
    _log("fluency: tiers A=%d B=%d C=%d docs (%.1fs, finished at %.1fs)"
         % (int(a_pos.size), int(b_pos.size), int(c_pos.size),
            time.time() - tf0, time.time() - t0))
    return {"order": order, "tier": tier, "thr": thr}


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
    tiers = None
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
        _log("classifier: scored %d survivors by %.1fs"
             % (len(survivors), time.time() - t0))

        # Fluency tiering (affects the fill order only)
        try:
            tiers = _fluency_tiers(docs, target_sample, survivors, scores, t0)
        except Exception as e:
            tiers = None
            _log("fluency: failed (%r); tiering SKIPPED" % (e,))

        # Report only: score the removed docs if there is time left.
        if removed:
            try:
                removed_scores = _score(removed, REPORT_DEADLINE, True)
            except Exception:
                removed_scores = None
    except Exception:
        scores = None
        tiers = None

    if scores is None:
        # Classifier failed: keep the filtered pool in seeded random order
        order = list(range(len(survivors)))
        rng.shuffle(order)
        _log("fluency: classifier unavailable; tiering SKIPPED")
    elif tiers is not None and len(tiers["order"]) == len(survivors):
        order = tiers["order"]
    else:
        tiers = None
        order = np.argsort(-scores, kind="stable").tolist()

    # Fill the token budget from the top
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    sel_counts = {}
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
            sel_counts[i] = c
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
        if tiers is not None:
            tier_arr = tiers["tier"]
            _log("fluency: tiering ACTIVE, thr=%.4f; tier sizes and share "
                 "of the selection:" % tiers["thr"])
            for name, tid in (("A", 0), ("B", 1), ("C", 2)):
                idxs = [survivors[int(k)]
                        for k in np.flatnonzero(tier_arr == tid)]
                n_exact = sum(1 for i in idxs if i in exact_counts)
                tok = sum(_tok(i) for i in idxs)
                sel_docs = sum(1 for i in idxs if i in sel_counts)
                sel_tok = sum(sel_counts.get(i, 0) for i in idxs)
                _log("    tier %s: %d docs, ~%d tokens = %.3f T "
                     "(%d docs counted exactly, rest estimated); "
                     "selected: %d docs, %d tokens = %.3f T"
                     % (name, len(idxs), int(tok), tok / float(T), n_exact,
                        sel_docs, int(sel_tok), sel_tok / float(T)))
        else:
            _log("fluency: tiering NOT applied; fill used plain "
                 "classifier-score order")
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

    # Seeded shuffle
    random.Random(seed).shuffle(selected)
    return selected
