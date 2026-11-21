"""Curator: target-similarity filter with dedup, junk removal, a word-order filter
and a tokens-per-character damage demotion in the fill order.

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
  5. Tokenise every survivor once (tiktoken gpt2, full text) and compute
     tpc = tokens / characters. Do the same for the target sample and set
     tpc_thr = 97.5th percentile of the target tpc values. A survivor with
     tpc > tpc_thr is marked 'damaged' (character-corrupted words and mojibake
     split into more BPE tokens). Damaged docs are demoted, never dropped.
  6. Fill the token budget T in three tiers, each by score descending:
       tier 1: not damaged, score >= 0.5
       tier 2: damaged,     score >= 0.5
       tier 3: all remaining docs
     The stored token counts are reused, so nothing is tokenised twice.
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
  - The up-front tokenisation of all survivors runs after the classifier has
    scored them (but before any fill decision). Running it before the
    classifier would eat into the scoring time budget and could shorten the
    scoring prefix, i.e. change the classifier scores, which must stay
    unchanged. Docs are tokenised in score-descending order, so if the time
    guard trips only the lowest-scored docs are left untokenised.
  - Docs not tokenised when the time guard trips are treated as not damaged;
    in the fill they are tokenised if time allows, otherwise their token count
    is estimated as len // 4 (the previous behaviour).
  - If tiktoken is unavailable or the target tpc cannot be calibrated (fewer
    than 20 non-empty target docs), no doc is marked damaged.
  - If the classifier failed (no scores), there are no tiers: the fill uses
    the seeded random order as before.

Diagnostics (printed to stderr, prefix "[curate]"):
  - target held-out rate percentiles (5/50/95) and the threshold
  - histogram of the word-order rate over the pool docs that passed dedup+junk
  - docs and tokens removed by the word-order filter
  - how many removed docs had classifier score >= 0.5 (removed docs are scored
    for the report only, after the kept docs, and only while time allows)
  - target tpc percentiles (5/50/97.5) and tpc_thr
  - histogram of tpc over survivors with score >= 0.5
  - docs and tokens (as a fraction of T) in each of the three tiers
  - docs and tokens of each tier that ended up in the selection
  - tokens (as a fraction of T) of remaining docs with score >= 0.5
  Token figures in the report are exact for tokenised docs and estimated from
  the measured tokens-per-char ratio for the rest.
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
TPC_PCT = 97.5         # percentile of target tokens-per-char used as threshold
SCORE_CUT = 0.5        # classifier score separating tiers 1/2 from tier 3
TOK_BATCH = 512        # docs per tokenisation batch
TOK_THREADS = 8
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

    n_surv = len(survivors)
    if scores is None:
        # Classifier failed: keep the filtered pool in seeded random order
        order = list(range(n_surv))
        rng.shuffle(order)
    else:
        order = np.argsort(-scores, kind="stable").tolist()

    # Tokeniser
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    # ------------------------------------------------------------------
    # Tokens-per-character: target calibration + up-front tokenisation
    # ------------------------------------------------------------------
    # All arrays below are indexed by position in `survivors`.
    tok_counts = np.full(n_surv, -1, dtype=np.int64)   # -1 = not tokenised
    char_counts = np.array([len(docs[i]) for i in survivors], dtype=np.int64)
    target_tpc = None
    tpc_thr = None
    upfront_done = 0
    upfront_complete = False
    tk0 = time.time()

    if enc is not None:
        try:
            tt = [t for t in target_sample if len(t) > 0]
            if tt:
                ttoks = enc.encode_ordinary_batch(tt, num_threads=TOK_THREADS)
                vals = [len(k) / len(t) for k, t in zip(ttoks, tt)]
                del ttoks
                if len(vals) >= 20:
                    target_tpc = np.array(vals, dtype=np.float64)
                    tpc_thr = float(np.percentile(target_tpc, TPC_PCT))
        except Exception as e:
            target_tpc, tpc_thr = None, None
            _log("tpc: target tokenisation failed (%r)" % (e,))

        # Tokenise survivors in score-descending order so that, if the time
        # guard trips, only the lowest-scored docs remain untokenised.
        pos = 0
        stop_reason = None
        while pos < n_surv:
            if (time.time() - t0) >= TOKEN_DEADLINE:
                stop_reason = "time guard"
                break
            part = order[pos:pos + TOK_BATCH]
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[survivors[j]] for j in part],
                    num_threads=TOK_THREADS,
                )
            except Exception as e:
                stop_reason = "error %r" % (e,)
                break
            for j, k in zip(part, toks):
                tok_counts[j] = len(k)
            del toks
            upfront_done += len(part)
            pos += TOK_BATCH
        upfront_complete = upfront_done >= n_surv
        if stop_reason is not None:
            _log("tpc: up-front tokenisation stopped (%s) after %d of %d "
                 "survivors; the rest are treated as not damaged"
                 % (stop_reason, upfront_done, n_surv))
    else:
        _log("tpc: tiktoken unavailable; no damage demotion, token counts "
             "estimated as len // 4")
    _log("tpc: tokenised %d of %d survivors up front (%.1fs)"
         % (upfront_done, n_surv, time.time() - tk0))

    # tpc and damage flag (frozen here; docs tokenised later in the fill are
    # treated as not damaged)
    upfront_mask = tok_counts >= 0
    tpc = np.full(n_surv, np.nan, dtype=np.float64)
    if upfront_mask.any():
        tpc[upfront_mask] = (
            tok_counts[upfront_mask] / np.maximum(char_counts[upfront_mask], 1)
        )
    damaged = np.zeros(n_surv, dtype=bool)
    if tpc_thr is not None:
        damaged[upfront_mask] = tpc[upfront_mask] > tpc_thr

    # ------------------------------------------------------------------
    # Fill order: three tiers, each by score descending
    # ------------------------------------------------------------------
    tier = np.full(n_surv, 2, dtype=np.int64)
    if scores is not None:
        hi_mask = scores >= SCORE_CUT
        tier[hi_mask & ~damaged] = 0
        tier[hi_mask & damaged] = 1
        fill_order = (
            [j for j in order if tier[j] == 0]
            + [j for j in order if tier[j] == 1]
            + [j for j in order if tier[j] == 2]
        )
    else:
        hi_mask = np.zeros(n_surv, dtype=bool)
        fill_order = list(order)

    # Fill the token budget, reusing the stored token counts
    selected = []
    total = 0
    sel_docs = [0, 0, 0]
    sel_tok = [0, 0, 0]
    pos = 0
    while pos < n_surv and total < T:
        part = fill_order[pos:pos + TOK_BATCH]
        pos += TOK_BATCH
        missing = [j for j in part if tok_counts[j] < 0]
        if (missing and enc is not None
                and (time.time() - t0) < TOKEN_DEADLINE):
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[survivors[j]] for j in missing],
                    num_threads=TOK_THREADS,
                )
                for j, k in zip(missing, toks):
                    tok_counts[j] = len(k)
                del toks
            except Exception:
                pass
        for j in part:
            c = int(tok_counts[j])
            if c < 0:
                c = max(1, int(char_counts[j]) // 4)
            selected.append(survivors[j])
            total += c
            tj = int(tier[j])
            sel_docs[tj] += 1
            sel_tok[tj] += c
            if total >= T:
                break

    if not selected:
        selected = list(survivors)

    # Diagnostics
    try:
        exact_mask = tok_counts >= 0
        exact_tok = int(tok_counts[exact_mask].sum())
        exact_chars = int(char_counts[exact_mask].sum())
        ratio = (exact_tok / exact_chars) if exact_chars > 0 else 0.25
        est_tok = np.where(exact_mask, tok_counts, char_counts * ratio)

        _log("fill: selected %d docs, %d tokens (%.3f T); tokens/char=%.4f "
             "(measured on %d docs)"
             % (len(selected), total, total / float(T), ratio,
                int(exact_mask.sum())))

        if order_thr is not None:
            rem_tok = sum(len(docs[i]) * ratio for i in removed)
            _log("word-order: removed %d docs, ~%d tokens (est., %.3f T)"
                 % (len(removed), int(rem_tok), rem_tok / float(T)))
            if removed:
                if removed_scores is not None:
                    done = ~np.isnan(removed_scores)
                    n_done = int(done.sum())
                    n_hi = int((removed_scores[done] >= 0.5).sum())
                    hi_tok = sum(
                        len(docs[i]) * ratio
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

        # tpc calibration
        if target_tpc is not None:
            q5, q50, q975 = (
                float(x) for x in np.percentile(target_tpc, [5, 50, 97.5])
            )
            _log("tpc: target (n=%d) p5=%.4f p50=%.4f p97.5=%.4f -> thr=%.4f"
                 % (len(target_tpc), q5, q50, q975, tpc_thr))
        else:
            _log("tpc: target tpc not calibrated; no doc marked damaged")
        _log("tpc: up-front tokenisation %s (%d of %d survivors); "
             "damaged docs overall: %d"
             % ("complete" if upfront_complete else "INCOMPLETE",
                upfront_done, n_surv, int(damaged.sum())))

        if scores is None:
            _log("tpc: classifier unavailable; no tiers, fill used the "
                 "seeded random order (all docs reported as tier 3)")
        else:
            # Histogram of tpc over survivors with score >= 0.5
            hmask = hi_mask & upfront_mask
            vals = tpc[hmask]
            if vals.size > 0:
                lo_v, hi_v = float(vals.min()), float(vals.max())
                if tpc_thr is not None:
                    lo_v = min(lo_v, tpc_thr)
                    hi_v = max(hi_v, tpc_thr)
                hi_v = hi_v + max(1e-6, 1e-6 * abs(hi_v))
                hist, edges = np.histogram(vals, bins=20, range=(lo_v, hi_v))
                _log("tpc: histogram over %d survivors with score >= %.1f "
                     "(%d more not tokenised up front):"
                     % (int(vals.size), SCORE_CUT,
                        int((hi_mask & ~upfront_mask).sum())))
                for h, lo, up in zip(hist, edges[:-1], edges[1:]):
                    mark = ""
                    if tpc_thr is not None and lo <= tpc_thr < up:
                        mark = " <- thr"
                    _log("    [%.4f, %.4f) %7d%s" % (lo, up, int(h), mark))
            else:
                _log("tpc: no tokenised survivors with score >= %.1f; "
                     "no histogram" % SCORE_CUT)

        # Tier report
        names = [
            "tier 1 (clean, score >= %.1f)" % SCORE_CUT,
            "tier 2 (damaged, score >= %.1f)" % SCORE_CUT,
            "tier 3 (remaining)",
        ]
        for k in range(3):
            m = tier == k
            n_k = int(m.sum())
            tok_k = float(est_tok[m].sum()) if n_k else 0.0
            n_exact = int((m & exact_mask).sum())
            _log("%s: %d docs, ~%d tokens = %.3f T (%d counted exactly); "
                 "selected %d docs, %d tokens = %.3f T"
                 % (names[k], n_k, int(tok_k), tok_k / float(T), n_exact,
                    sel_docs[k], sel_tok[k], sel_tok[k] / float(T)))

        if scores is not None:
            n_hi = int(hi_mask.sum())
            hi_tok = float(est_tok[hi_mask].sum()) if n_hi else 0.0
            n_exact = int((hi_mask & exact_mask).sum())
            _log("remaining docs with score >= 0.5: %d docs, ~%d tokens "
                 "= %.3f T (%d docs counted exactly, rest estimated)"
                 % (n_hi, int(hi_tok), hi_tok / float(T), n_exact))
        _log("total time %.1fs" % (time.time() - t0))
    except Exception as e:
        _log("report failed: %r" % (e,))

    # Seeded shuffle
    random.Random(seed).shuffle(selected)
    return selected
