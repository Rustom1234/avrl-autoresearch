"""Curator: target-similarity filter with dedup, junk removal and a word-order filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop exact / near duplicates (hash of normalised first 500 chars, keep first).
  2. Drop junk (too short, low unique-word ratio, low alphabetic/space share).
  3. Word-order rate for every dedup+junk survivor, over the FULL text:
       - words = lower-cased whitespace split of the whole doc
       - B = adjacent word bigrams occurring >= 2 times across the full text of
         all target_sample docs
       - rate(doc) = fraction of the doc's adjacent word pairs that are in B
       - calibration: 10-fold cross-fit over target_sample (folds drawn from
         the seed + 7919 stream); each held-out doc is scored on its full text
         against the bigram set (count >= 2) built from the other 9 folds.
  4. Score ALL survivors with a char n-gram TF-IDF + logistic regression
     (target_sample = positives, seeded random pool sample = negatives).
  5. Tokenise (or estimate) the survivors' tokens, then choose the word-order
     threshold adaptively: candidates are the 10th, 5th and 2nd percentile of
     the held-out target rates (no factor). The strictest candidate for which
     the docs with rate >= thr AND classifier score >= 0.5 still hold at least
     1.05 T tokens is chosen; if none qualifies, the 2nd percentile is used.
  6. Drop survivors with rate < chosen thr.
  7. Take the best scored remaining docs until the token budget T is filled.
  8. Shuffle the selection with the CURATE_SEED seed.

Time guards / deviations from the written spec:
  - Word-order pass: every 256 docs the end of the pass is projected from the
    measured seconds-per-char. If it would run beyond 45 s since the start,
    the REMAINING docs are scored on their first 12000 chars and the target
    calibration (all three candidates) is done on the same 12000-char window.
    Docs scored before the switch keep their full-text rate (identical for
    docs up to 12000 chars).
  - The vectorizer is fitted on the 3200 training docs (targets + negatives),
    not on the whole pool.
  - The prefix used to score pool docs shrinks below 2000 chars if time is short.
  - Junk statistics are computed on the first 3000 chars of each doc.
  - The junk thresholds (0.2 unique-word ratio, 0.8 alphabetic/space share) are
    loosened if the target sample itself falls below them.
  - Tokens: docs with score >= 0.5 and rate >= the loosest candidate are
    tokenised exactly, best score first, until 150 s; anything not tokenised
    is estimated from the measured tokens-per-char ratio. The candidate
    check therefore uses exact counts where available and estimates otherwise.
  - If the classifier is unavailable, no candidate can be checked and the
    2nd percentile is used.
  - The word-order filter is skipped if it cannot be calibrated (fewer than 20
    usable held-out target rates) or if it would remove every surviving doc.
  - The comparison with the current rule (first 3000 chars, 10-fold, 2nd
    percentile) is for the report only; its extra pass is abandoned after 70 s.

Diagnostics (printed to stderr, prefix "[curate]"):
  - target held-out rate percentiles (2/5/10/50) on the full text (and on the
    12000-char window if the time guard fired)
  - histogram of the word-order rate over the pool docs that passed dedup+junk
  - for each candidate threshold: docs and tokens (fraction of T) removed, and
    tokens of the remaining docs with score >= 0.5; whether it qualifies
  - which candidate was chosen
  - how many docs kept by the current rule (first 3000 chars, 2nd percentile)
    are now removed (and how many of those had classifier score >= 0.5)
  - fill size, and a WARNING if remaining docs with score >= 0.5 hold < T
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
ORDER_WINDOW = None    # chars used for the word-order rate (None = full text)
ORDER_FALLBACK_WINDOW = 12000   # window used if the rate pass runs late
ORDER_DEADLINE = 45.0  # seconds since start by which the rate pass should end
ORDER_REPORT_DEADLINE = 70.0    # current-rule comparison pass stops here
ORDER_MIN_COUNT = 2    # min occurrences of a bigram in the target sample
ORDER_FOLDS = 10       # cross-fit folds for the threshold calibration
ORDER_PCTS = (10.0, 5.0, 2.0)   # candidate percentiles, strictest first
ORDER_MARGIN = 1.05    # required tokens (x T) of kept docs with score >= 0.5
CUR_WINDOW = 3000      # current rule (report only): first 3000 chars
CUR_PCT = 2.0          # current rule (report only): 2nd percentile
FEATURE_DEADLINE = 100.0   # seconds since start by which scoring should finish
PRETOK_DEADLINE = 150.0    # tokenising for the threshold choice stops here
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


def _order_rate(text, bset, exclude=None, window=None):
    """Fraction of adjacent word pairs in bset (full text or first `window` chars).

    Pairs in `exclude` are treated as not being in bset.
    Returns None if the text has fewer than two words.
    """
    s = text if window is None else text[:window]
    w = s.lower().split()
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

    _log("dedup+junk: %d of %d docs survive (%.1fs)"
         % (len(survivors), n_docs, time.time() - t0))

    # ------------------------------------------------------------------
    # Word-order rates for all survivors + candidate thresholds
    # ------------------------------------------------------------------
    rates = None        # word-order rate per survivor (new rule)
    cur_rates = None    # rate per survivor under the current rule (report)
    cur_thr = None      # threshold of the current rule (report)
    cands = None        # list of (percentile, threshold), strictest first
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

        n_folds = min(ORDER_FOLDS, len(perm))
        fold_info = []
        if n_folds >= 2:
            for k in range(n_folds):
                fold = perm[k::n_folds]
                fold_cnt = Counter()
                for i in fold:
                    fold_cnt.update(bg[i])
                # bigrams that fall below the min count without this fold
                excl = {b for b, c in fold_cnt.items()
                        if total_cnt[b] - c < ORDER_MIN_COUNT}
                fold_info.append((fold, excl))
        del bg, total_cnt

        def _held(window):
            out = []
            for fold, excl in fold_info:
                for i in fold:
                    r = _order_rate(tgt[i], full_set, excl, window)
                    if r is not None:
                        out.append(r)
            return np.array(out, dtype=np.float64)

        held_full = _held(ORDER_WINDOW)

        if len(held_full) >= 20:
            p2, p5, p10, p50 = (
                float(x) for x in np.percentile(held_full, [2, 5, 10, 50])
            )
            _log("word-order: target held-out rate, FULL text (%d-fold, n=%d) "
                 "p2=%.4f p5=%.4f p10=%.4f p50=%.4f"
                 % (n_folds, len(held_full), p2, p5, p10, p50))

            # Pool pass: full text, with a projected-time guard.
            n_s = len(survivors)
            rates_arr = np.zeros(n_s, dtype=np.float64)
            cur_arr = np.full(n_s, np.nan, dtype=np.float64)
            total_chars = sum(len(docs[i]) for i in survivors)
            done_chars = 0
            window = ORDER_WINDOW
            switch_at = None
            do_cur = True
            tp0 = time.time()
            for k, i in enumerate(survivors):
                if k % 256 == 0:
                    now = time.time()
                    el = now - t0
                    if switch_at is None:
                        if done_chars > 0:
                            proj = el + (now - tp0) / done_chars * (
                                total_chars - done_chars)
                        else:
                            proj = el
                        if proj > ORDER_DEADLINE:
                            window = ORDER_FALLBACK_WINDOW
                            switch_at = k
                    if do_cur and el > ORDER_REPORT_DEADLINE:
                        do_cur = False
                d = docs[i]
                s = d if window is None else d[:window]
                w = s.lower().split()
                n = len(w) - 1
                if n > 0:
                    r = sum(1 for b in zip(w, w[1:]) if b in full_set) / n
                else:
                    r = 0.0
                rates_arr[k] = r
                if switch_at is None:
                    done_chars += len(d)
                # current rule (first CUR_WINDOW chars), report only
                if len(d) <= CUR_WINDOW:
                    cur_arr[k] = r
                elif do_cur:
                    w3 = d[:CUR_WINDOW].lower().split()
                    n3 = len(w3) - 1
                    if n3 > 0:
                        cur_arr[k] = sum(
                            1 for b in zip(w3, w3[1:]) if b in full_set) / n3
                    else:
                        cur_arr[k] = 0.0

            if switch_at is None:
                held_used = held_full
                _log("word-order: pool rates on the FULL text for all %d "
                     "docs (%.1fs, done at %.1fs)"
                     % (n_s, time.time() - tw0, time.time() - t0))
            else:
                held_used = _held(ORDER_FALLBACK_WINDOW)
                q2, q5, q10, q50 = (
                    float(x) for x in np.percentile(held_used, [2, 5, 10, 50])
                ) if len(held_used) else (0.0, 0.0, 0.0, 0.0)
                _log("word-order: TIME GUARD fired at doc %d of %d; the "
                     "remaining %d docs use the first %d chars and the "
                     "target is calibrated on the same window (done at %.1fs)"
                     % (switch_at, n_s, n_s - switch_at,
                        ORDER_FALLBACK_WINDOW, time.time() - t0))
                _log("word-order: target held-out rate, first %d chars "
                     "(n=%d) p2=%.4f p5=%.4f p10=%.4f p50=%.4f"
                     % (ORDER_FALLBACK_WINDOW, len(held_used),
                        q2, q5, q10, q50))

            if len(held_used) >= 20:
                rates = rates_arr
                cands = [(p, float(np.percentile(held_used, p)))
                         for p in ORDER_PCTS]
                _log("word-order: candidate thresholds "
                     + ", ".join("p%g=%.4f" % (p, t) for p, t in cands))

                # Current rule threshold (report only)
                try:
                    held_cur = _held(CUR_WINDOW)
                    if len(held_cur) >= 20:
                        cur_thr = float(np.percentile(held_cur, CUR_PCT))
                        cur_rates = cur_arr
                        _log("word-order: current rule (first %d chars, "
                             "p%g) thr = %.4f"
                             % (CUR_WINDOW, CUR_PCT, cur_thr))
                except Exception as e:
                    cur_thr = None
                    cur_rates = None
                    _log("word-order: current-rule comparison failed (%r)"
                         % (e,))

                # Pool histogram (docs that passed dedup + junk)
                top_thr = max(t for _, t in cands)
                hi_edge = max(float(rates.max()), top_thr * 2.0, 1e-6)
                hist, edges = np.histogram(rates, bins=20,
                                           range=(0.0, hi_edge))
                _log("word-order: pool rate histogram over %d docs "
                     "(after dedup+junk):" % n_s)
                for h, lo, up in zip(hist, edges[:-1], edges[1:]):
                    mark = ""
                    for p, t in cands:
                        if lo <= t < up:
                            mark += " <- p%g" % p
                    _log("    [%.4f, %.4f) %7d%s" % (lo, up, int(h), mark))
            else:
                _log("word-order: only %d usable held-out target rates on "
                     "the fallback window; filter skipped" % len(held_used))
        else:
            _log("word-order: only %d usable held-out target rates; "
                 "filter skipped" % len(held_full))
        del full_set
    except Exception as e:  # never let the filter break curation
        rates = None
        cands = None
        cur_rates = None
        cur_thr = None
        _log("word-order: rate pass failed (%r); filter skipped" % (e,))

    # ------------------------------------------------------------------
    # Classifier: score ALL dedup+junk survivors (before the drop)
    # ------------------------------------------------------------------
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

        def _score(idx_list, deadline):
            out = np.full(len(idx_list), np.nan, dtype=np.float64)
            chunk = 2000
            for start in range(0, len(idx_list), chunk):
                part = idx_list[start:start + chunk]
                remaining_docs = len(idx_list) - start
                remaining_time = deadline - (time.time() - t0)
                if st["chars"] > 0:
                    st["rate"] = st["time"] / st["chars"]
                if remaining_time <= 0:
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

        scores = _score(survivors, FEATURE_DEADLINE)
        _log("classifier: scored %d docs (done at %.1fs)"
             % (len(survivors), time.time() - t0))
    except Exception as e:
        scores = None
        _log("classifier failed (%r)" % (e,))

    # ------------------------------------------------------------------
    # Tokeniser + exact counts for the docs that matter for the choice
    # ------------------------------------------------------------------
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    tok_exact = {}   # doc index -> exact token count

    def _tokenise(idx_list, deadline):
        if enc is None:
            return
        for start in range(0, len(idx_list), 512):
            if (time.time() - t0) >= deadline:
                break
            part = [i for i in idx_list[start:start + 512]
                    if i not in tok_exact]
            if not part:
                continue
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in part], num_threads=8
                )
            except Exception:
                break
            for i, t in zip(part, toks):
                tok_exact[i] = len(t)

    def _tpc():
        if not tok_exact:
            return 0.25
        ch = sum(len(docs[i]) for i in tok_exact)
        return (sum(tok_exact.values()) / ch) if ch > 0 else 0.25

    # ------------------------------------------------------------------
    # Adaptive threshold choice and drop
    # ------------------------------------------------------------------
    removed_n = 0
    chosen = None
    if cands is not None and rates is not None:
        try:
            n_s = len(survivors)
            loose_thr = min(t for _, t in cands)
            hi = None
            if scores is not None:
                hi = scores >= 0.5
                need = np.nonzero(hi & (rates >= loose_thr))[0]
                need = need[np.argsort(-scores[need], kind="stable")]
                _tokenise([survivors[k] for k in need.tolist()],
                          PRETOK_DEADLINE)
                _log("tokens: %d of %d docs with score >= 0.5 and rate >= "
                     "p2 tokenised exactly (done at %.1fs)"
                     % (sum(1 for k in need.tolist()
                            if survivors[k] in tok_exact),
                        len(need), time.time() - t0))
            tpc = _tpc()
            tok_est = np.empty(n_s, dtype=np.float64)
            for k, i in enumerate(survivors):
                c = tok_exact.get(i)
                tok_est[k] = c if c is not None else len(docs[i]) * tpc
            _log("tokens: tokens/char=%.4f (measured on %d docs); pool after "
                 "dedup+junk ~%.3f T"
                 % (tpc, len(tok_exact), tok_est.sum() / float(T)))

            for p, thr in cands:
                mask = rates >= thr
                rem_docs = int((~mask).sum())
                rem_tok = float(tok_est[~mask].sum())
                if hi is not None:
                    hi_tok = float(tok_est[mask & hi].sum())
                    ok = hi_tok >= ORDER_MARGIN * T
                    _log("word-order: candidate p%g thr=%.4f: removes %d "
                         "docs, ~%.3f T tokens; remaining docs with score "
                         ">= 0.5: %d docs, ~%.3f T -> %s (need %.2f T)"
                         % (p, thr, rem_docs, rem_tok / float(T),
                            int((mask & hi).sum()), hi_tok / float(T),
                            "QUALIFIES" if ok else "does not qualify",
                            ORDER_MARGIN))
                    if ok and chosen is None:
                        chosen = (p, thr)
                else:
                    _log("word-order: candidate p%g thr=%.4f: removes %d "
                         "docs, ~%.3f T tokens; remaining score >= 0.5 "
                         "tokens: n/a (classifier unavailable)"
                         % (p, thr, rem_docs, rem_tok / float(T)))

            if chosen is None:
                chosen = cands[-1]
                _log("word-order: no candidate qualifies%s; falling back to "
                     "p%g thr=%.4f"
                     % ("" if hi is not None else " (classifier unavailable)",
                        chosen[0], chosen[1]))
            else:
                _log("word-order: CHOSEN candidate p%g thr=%.4f"
                     % (chosen[0], chosen[1]))

            keep_mask = rates >= chosen[1]
            if keep_mask.any():
                # Comparison with the current rule (report only)
                try:
                    if cur_rates is not None and cur_thr is not None:
                        known = ~np.isnan(cur_rates)
                        cur_keep = known & (cur_rates >= cur_thr)
                        newly = cur_keep & ~keep_mask
                        back = known & ~cur_keep & keep_mask
                        msg = ("word-order: docs kept by the current rule "
                               "(first %d chars, p%g, thr=%.4f) that are NOW "
                               "REMOVED: %d docs, ~%.3f T tokens"
                               % (CUR_WINDOW, CUR_PCT, cur_thr,
                                  int(newly.sum()),
                                  float(tok_est[newly].sum()) / float(T)))
                        if hi is not None:
                            msg += ("; of those with score >= 0.5: %d docs, "
                                    "~%.3f T"
                                    % (int((newly & hi).sum()),
                                       float(tok_est[newly & hi].sum())
                                       / float(T)))
                        _log(msg)
                        _log("word-order: current rule kept %d docs, removed "
                             "%d; docs removed by the current rule but kept "
                             "now: %d"
                             % (int(cur_keep.sum()),
                                int((known & ~cur_keep).sum()),
                                int(back.sum())))
                        n_unknown = int((~known).sum())
                        if n_unknown:
                            _log("word-order: NOTE current-rule rate not "
                                 "computed for %d docs (report deadline); "
                                 "they are excluded from the comparison"
                                 % n_unknown)
                    else:
                        _log("word-order: comparison with the current rule "
                             "not available")
                except Exception as e:
                    _log("word-order: comparison report failed (%r)" % (e,))

                removed_n = int((~keep_mask).sum())
                rem_tok = float(tok_est[~keep_mask].sum())
                survivors = [i for i, m in zip(survivors, keep_mask) if m]
                if scores is not None:
                    scores = scores[keep_mask]
                _log("word-order: removed %d docs (~%.3f T tokens), kept %d"
                     % (removed_n, rem_tok / float(T), len(survivors)))
            else:
                _log("word-order: filter would remove everything; skipped")
        except Exception as e:  # never let the filter break curation
            _log("word-order: threshold choice failed (%r); filter skipped"
                 % (e,))

    # ------------------------------------------------------------------
    # Fill the token budget from the top
    # ------------------------------------------------------------------
    if scores is None or len(scores) != len(survivors):
        # Classifier failed: keep the filtered pool in seeded random order
        scores = None
        order = list(range(len(survivors)))
        rng.shuffle(order)
    else:
        order = np.argsort(-scores, kind="stable").tolist()

    selected = []
    total = 0
    batch = 512
    pos = 0
    n_surv = len(order)
    while pos < n_surv and total < T:
        part = [survivors[j] for j in order[pos:pos + batch]]
        pos += batch
        missing = [i for i in part if i not in tok_exact]
        if missing and enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in missing], num_threads=8
                )
                for i, t in zip(missing, toks):
                    tok_exact[i] = len(t)
            except Exception:
                pass
        for i in part:
            c = tok_exact.get(i)
            if c is None:
                c = max(1, len(docs[i]) // 4)
            selected.append(i)
            total += c
            if total >= T:
                break

    if not selected:
        selected = list(survivors)

    # Diagnostics
    try:
        tpc = _tpc()

        def _tok(i):
            c = tok_exact.get(i)
            return c if c is not None else len(docs[i]) * tpc

        _log("fill: selected %d docs, %d tokens (%.3f T); tokens/char=%.4f "
             "(measured on %d docs)"
             % (len(selected), total, total / float(T), tpc, len(tok_exact)))
        if total < T:
            _log("WARNING: the whole filtered pool holds only %.3f T tokens; "
                 "the trainer will cycle the selection" % (total / float(T)))
        if chosen is not None:
            _log("word-order: final threshold p%g = %.4f, %d docs removed"
                 % (chosen[0], chosen[1], removed_n))

        if scores is not None:
            hi_idx = [i for i, s in zip(survivors, scores) if s >= 0.5]
            n_exact = sum(1 for i in hi_idx if i in tok_exact)
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
