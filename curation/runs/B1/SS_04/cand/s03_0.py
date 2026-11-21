"""Curator: target-similarity filter with dedup, junk and damaged-text removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop exact / near duplicates (hash of normalised first 500 chars, keep first).
  2. Drop junk (too short, low unique-word ratio, low alphabetic/space share).
  3. Drop damaged text:
       a. out-of-vocabulary filter: V = words ([a-z]+ on the lower-cased first
          3000 chars) with pool document frequency >= 5, plus every word in
          target_sample. oov = share of a doc's words of length >= 4 (first
          3000 chars) not in V. Threshold = max(1.5 * p95 of the target docs'
          rates measured against the pool-only vocabulary, 0.02).
       b. mojibake filter: more than 1 marker per 1000 chars in the first
          3000 chars (markers: 'â€', 'Ã', 'Â', U+FFFD).
  4. Score surviving docs with a char n-gram TF-IDF + logistic regression
     (target_sample = positives, seeded random pool sample = negatives).
  5. Take the best scored docs until the token budget T is filled.
  6. Shuffle the selection with the CURATE_SEED seed.

Reporting: after each call a one-line JSON report is printed to stderr and kept
in the module-level LAST_REPORT. It gives the docs and tokens removed by each
damaged-text filter and the tokens (also as a fraction of T) of remaining docs
with score >= 0.5. Token counts are exact when time allows; otherwise the rest
is estimated from characters and "tokens_exact" is false.

Deviations from the written spec, made to stay inside the 180 s limit or to
avoid discarding target-like text:
  - The vectorizer is fitted on the 3200 training docs (targets + negatives),
    not on the whole pool.
  - The prefix used to score pool docs shrinks below 2000 chars if time is short.
  - Junk statistics are computed on the first 3000 chars of each doc, so long
    documents are not penalised for a naturally lower unique-word ratio.
  - The junk thresholds (0.2 unique-word ratio, 0.8 alphabetic/space share) are
    loosened if the target sample itself falls below them.

Notes on the damaged-text filter:
  - Only words of length >= 4 are ever looked up, so the vocabulary stores only
    those (regex [a-z]{4,} gives exactly the [a-z]+ words of length >= 4).
  - Document frequency is counted over the whole pool as specified, duplicates
    included, so a damaged doc present 5+ times puts its own typos into V.
  - The classifier's negatives are still sampled from the whole pool, as before.
"""
import hashlib
import json
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
DAMAGE_WINDOW = 3000   # chars used for the OOV and mojibake statistics
MIN_DF = 5             # pool document frequency needed for a word to be in V
OOV_FLOOR = 0.02       # lower bound of the OOV threshold
OOV_MULT = 1.5         # multiplier on the target 95th percentile
MOJI_PER_1000 = 1.0    # max mojibake markers per 1000 chars
FEATURE_DEADLINE = 100.0   # seconds since start by which scoring should finish
TOKEN_DEADLINE = 160.0     # seconds since start after which we stop tokenising
REPORT_DEADLINE = 160.0    # seconds since start after which the report estimates

_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")
_LONG_WORD = re.compile(r"[a-z]{4,}")
# 'â€', 'Ã', 'Â', replacement character
_MOJI_MARKERS = ("\u00e2\u20ac", "\u00c3", "\u00c2", "\ufffd")

LAST_REPORT = {}


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


def _long_words(text):
    """Words ([a-z]+) of length >= 4 in the lower-cased damage window."""
    return _LONG_WORD.findall(text[:DAMAGE_WINDOW].lower())


def _oov_rate(words, vocab):
    if not words:
        return 0.0
    miss = 0
    for w in words:
        if w not in vocab:
            miss += 1
    return miss / len(words)


def _moji_rate(text):
    """Mojibake markers per 1000 chars of the damage window."""
    s = text[:DAMAGE_WINDOW]
    n = len(s)
    if n == 0:
        return 0.0
    c = 0
    for m in _MOJI_MARKERS:
        c += s.count(m)
    return 1000.0 * c / n


def _emit_report(report):
    global LAST_REPORT
    LAST_REPORT = report
    try:
        print("[curate] " + json.dumps(report, sort_keys=True), file=sys.stderr)
        sys.stderr.flush()
    except Exception:
        pass


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

    # Damaged-text filter, part 1: vocabulary and OOV threshold
    df = Counter()
    for d in docs:
        df.update(set(_long_words(d)))
    pool_vocab = {w for w, c in df.items() if c >= MIN_DF}
    del df

    target_rates = []
    for t in target_sample:
        tw = _long_words(t)
        if tw:
            target_rates.append(_oov_rate(tw, pool_vocab))
    target_p95 = float(np.quantile(np.array(target_rates), 0.95)) if target_rates else 0.0
    oov_thr = max(OOV_MULT * target_p95, OOV_FLOOR)

    vocab = set(pool_vocab)
    for t in target_sample:
        vocab.update(_LONG_WORD.findall(t.lower()))
    del pool_vocab

    # Dedup, junk filter, then the two damaged-text filters
    seen = set()
    survivors = []
    oov_removed = []     # indices flagged by the OOV filter
    moji_removed = []    # indices flagged by the mojibake filter
    n_both = 0
    n_clean_before_damage = 0
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
        n_clean_before_damage += 1
        bad_oov = _oov_rate(_long_words(d), vocab) > oov_thr
        bad_moji = _moji_rate(d) > MOJI_PER_1000
        if bad_oov:
            oov_removed.append(i)
        if bad_moji:
            moji_removed.append(i)
        if bad_oov and bad_moji:
            n_both += 1
        if bad_oov or bad_moji:
            continue
        survivors.append(i)
    del seen
    del vocab

    report = {
        "pool_docs": n_docs,
        "docs_after_dedup_and_junk": n_clean_before_damage,
        "target_oov_p95": round(target_p95, 6),
        "oov_threshold": round(oov_thr, 6),
        "oov_removed_docs": len(oov_removed),
        "mojibake_removed_docs": len(moji_removed),
        "removed_by_both_docs": n_both,
        "damaged_removed_docs_total": len(oov_removed) + len(moji_removed) - n_both,
        "docs_remaining": len(survivors),
        "T": int(T),
    }

    if not survivors:
        report["note"] = "no survivors; returning the whole pool shuffled"
        _emit_report(report)
        idx = list(range(n_docs))
        rng.shuffle(idx)
        return idx

    # Classifier
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
        rate = fit_time / train_chars  # seconds per char (pessimistic)

        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
        )
        clf.fit(Xtr, y)
        pos_col = list(clf.classes_).index(1)

        scores = np.zeros(len(survivors), dtype=np.float64)
        chunk = 2000
        done_chars, done_time = 0, 0.0
        for start in range(0, len(survivors), chunk):
            part = survivors[start:start + chunk]
            remaining_docs = len(survivors) - start
            remaining_time = FEATURE_DEADLINE - (time.time() - t0)
            if done_chars > 0:
                rate = done_time / done_chars
            if remaining_time <= 0:
                L = 200
            else:
                L = int(remaining_time / (rate * remaining_docs + 1e-12))
                L = max(200, min(PREFIX, L))
            texts = [docs[i][:L] for i in part]
            tc = time.time()
            X = vec.transform(texts)
            scores[start:start + len(part)] = clf.predict_proba(X)[:, pos_col]
            done_time += time.time() - tc
            done_chars += sum(len(t) for t in texts)
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

    tok_cache = {}   # doc index -> exact token count (only exact counts)
    selected = []
    total = 0
    batch = 512
    pos = 0
    n_surv = len(order)
    while pos < n_surv and total < T:
        part = [survivors[j] for j in order[pos:pos + batch]]
        pos += batch
        counts = None
        if enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in part], num_threads=8
                )
                counts = [len(t) for t in toks]
                for i, c in zip(part, counts):
                    tok_cache[i] = c
            except Exception:
                counts = None
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in part]
        for i, c in zip(part, counts):
            selected.append(i)
            total += c
            if total >= T:
                break

    if not selected:
        selected = list(survivors)

    # Report (never allowed to affect the selection)
    try:
        if scores is not None:
            high = [survivors[j] for j in np.nonzero(scores >= 0.5)[0].tolist()]
        else:
            high = None

        need = set(oov_removed) | set(moji_removed)
        if high is not None:
            need |= set(high)
        missing = [i for i in need if i not in tok_cache]
        missing.sort()
        rb = 256
        p = 0
        while (
            enc is not None
            and p < len(missing)
            and (time.time() - t0) < REPORT_DEADLINE
        ):
            part = missing[p:p + rb]
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in part], num_threads=8
                )
            except Exception:
                break
            for i, tk in zip(part, toks):
                tok_cache[i] = len(tk)
            p += rb

        if tok_cache:
            c_chars = sum(len(docs[i]) for i in tok_cache) or 1
            ratio = sum(tok_cache.values()) / c_chars
        else:
            ratio = 0.25
        exact = all(i in tok_cache for i in need)

        def ntok(indices):
            s = 0.0
            for i in indices:
                c = tok_cache.get(i)
                s += c if c is not None else len(docs[i]) * ratio
            return int(round(s))

        oov_set = set(oov_removed)
        union = oov_removed + [i for i in moji_removed if i not in oov_set]
        report["oov_removed_tokens"] = ntok(oov_removed)
        report["mojibake_removed_tokens"] = ntok(moji_removed)
        report["damaged_removed_tokens_total"] = ntok(union)
        report["oov_removed_tokens_frac_T"] = round(report["oov_removed_tokens"] / T, 4)
        report["mojibake_removed_tokens_frac_T"] = round(
            report["mojibake_removed_tokens"] / T, 4
        )
        if high is not None:
            ht = ntok(high)
            report["score_ge_0.5_docs"] = len(high)
            report["score_ge_0.5_tokens"] = ht
            report["score_ge_0.5_tokens_frac_T"] = round(ht / T, 4)
        else:
            report["score_ge_0.5_docs"] = None
            report["score_ge_0.5_tokens"] = None
            report["score_ge_0.5_tokens_frac_T"] = None
            report["note"] = "classifier failed; no scores"
        report["tokens_exact"] = bool(exact)
        report["selected_docs"] = len(selected)
        report["selected_tokens_counted"] = int(total)
        report["selected_tokens_frac_T"] = round(total / T, 4)
        report["seconds"] = round(time.time() - t0, 1)
    except Exception as e:  # pragma: no cover - reporting must not break select
        report["report_error"] = repr(e)
    _emit_report(report)

    # Seeded shuffle
    random.Random(seed).shuffle(selected)
    return selected
