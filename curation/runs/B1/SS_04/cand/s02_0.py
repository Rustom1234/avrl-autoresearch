"""Curator: target-similarity filter with dedup and junk removal, training only
on the confidently target-like set (repeated in epochs if it is smaller than T).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop exact / near duplicates (hash of normalised first 500 chars, keep first).
  2. Drop junk (too short, low unique-word ratio, low alphabetic/space share).
  3. Score surviving docs with a char n-gram TF-IDF + logistic regression
     (target_sample = positives, seeded random pool sample = negatives).
  4. High-confidence set H = survivors with predict_proba >= 0.5.
     Count tokens(H) with tiktoken.
       a. tokens(H) >= T: take the best scored docs until the token budget T
          is filled, then one seeded shuffle (previous behaviour).
       b. tokens(H) <  T: do NOT pad with lower-scored docs. Use all of H
          (if H is under 0.2*T tokens, extend it with the next best scored
          docs until it reaches 0.2*T). Return consecutive epochs of that set,
          each epoch a fresh shuffle from the CURATE_SEED-seeded RNG, appended
          until the cumulative token count reaches T.
  5. Report tokens(H)/T and the number of docs in H on stderr (line starting
     with "[curate]"); the same figures are stored in the module-level
     LAST_REPORT dict.

Deviations from the written spec, made to stay inside the 180 s limit or to
avoid discarding target-like text:
  - The vectorizer is fitted on the 3200 training docs (targets + negatives),
    not on the whole pool.
  - The prefix used to score pool docs shrinks below 2000 chars if time is short.
  - Junk statistics are computed on the first 3000 chars of each doc, so long
    documents are not penalised for a naturally lower unique-word ratio.
  - The junk thresholds (0.2 unique-word ratio, 0.8 alphabetic/space share) are
    loosened if the target sample itself falls below them.
  - Token counts fall back to len(doc)//4 if tiktoken is unavailable or the
    tokenising deadline is passed; the report then says tokens_exact=False.
  - If the classifier fails there is no H; the filtered pool is used in seeded
    random order until T tokens (previous fallback behaviour).
"""
import hashlib
import os
import random
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

PREFIX = 2000          # chars used for the classifier features
JUNK_WINDOW = 3000     # chars used for junk statistics
FEATURE_DEADLINE = 100.0   # seconds since start by which scoring should finish
TOKEN_DEADLINE = 160.0     # seconds since start after which we stop tokenising
CONF_THRESHOLD = 0.5       # predict_proba threshold defining the set H
MIN_SET_FRACTION = 0.2     # minimum size of the repeated set, as a fraction of T

LAST_REPORT = {}

_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")


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


def _report(info):
    LAST_REPORT.clear()
    LAST_REPORT.update(info)
    try:
        parts = []
        for k, v in info.items():
            if isinstance(v, float):
                parts.append("%s=%.4f" % (k, v))
            else:
                parts.append("%s=%s" % (k, v))
        print("[curate] " + " ".join(parts), file=sys.stderr, flush=True)
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
        _report({"status": "no_survivors", "n_docs": n_docs})
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
    n_surv = len(order)

    # Token counting (lazy, in score order, with a deadline)
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    state = {"exact": True}
    batch = 512

    def count_tokens(doc_ids):
        """Token counts for a list of doc indices (>= 1 each)."""
        counts = None
        if enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in doc_ids], num_threads=8
                )
                counts = [max(1, len(t)) for t in toks]
            except Exception:
                counts = None
        if counts is None:
            state["exact"] = False
            counts = [max(1, len(docs[i]) // 4) for i in doc_ids]
        return counts

    # counts_by_rank[r] = tokens of survivors[order[r]], filled as a prefix
    counts_by_rank = []

    def ensure_counted(upto):
        upto = min(upto, n_surv)
        while len(counts_by_rank) < upto:
            start = len(counts_by_rank)
            end = min(upto, start + batch)
            part = [survivors[j] for j in order[start:end]]
            counts_by_rank.extend(count_tokens(part))

    # Fallback when there are no scores: old behaviour
    if scores is None:
        selected = []
        total = 0
        pos = 0
        while pos < n_surv and total < T:
            ensure_counted(pos + batch)
            end = min(n_surv, pos + batch)
            for r in range(pos, end):
                selected.append(survivors[order[r]])
                total += counts_by_rank[r]
                if total >= T:
                    break
            pos = end
        if not selected:
            selected = list(survivors)
        random.Random(seed).shuffle(selected)
        _report({
            "status": "classifier_failed",
            "n_survivors": n_surv,
            "n_selected": len(selected),
            "tokens_selected_over_T": total / float(T),
            "tokens_exact": state["exact"],
        })
        return selected

    # High-confidence set H = prefix of the score-sorted order
    sorted_scores = scores[np.asarray(order, dtype=np.int64)]
    n_H = int(np.sum(sorted_scores >= CONF_THRESHOLD))

    ensure_counted(n_H)
    tok_H = int(sum(counts_by_rank[:n_H]))

    info = {
        "status": "ok",
        "tokens_H_over_T": tok_H / float(T),
        "n_docs_H": n_H,
        "tokens_H": tok_H,
        "n_survivors": n_surv,
        "n_pool": n_docs,
    }

    if tok_H >= T:
        # Enough confident text: top-scored docs until T, one seeded shuffle
        selected = []
        total = 0
        for r in range(n_H):
            selected.append(survivors[order[r]])
            total += counts_by_rank[r]
            if total >= T:
                break
        n_used = len(selected)
        random.Random(seed).shuffle(selected)
        info.update({
            "mode": "top_until_T",
            "n_docs_used": n_used,
            "tokens_used_over_T": total / float(T),
            "epochs": 1.0,
            "n_returned": len(selected),
            "tokens_exact": state["exact"],
            "seconds": time.time() - t0,
        })
        _report(info)
        return selected

    # Not enough confident text: repeat H instead of padding
    n_use = n_H
    tok_use = tok_H
    min_tokens = MIN_SET_FRACTION * T
    extended = 0
    while tok_use < min_tokens and n_use < n_surv:
        ensure_counted(n_use + batch)
        end = min(n_surv, n_use + batch)
        while n_use < end and tok_use < min_tokens:
            tok_use += counts_by_rank[n_use]
            n_use += 1
            extended += 1

    if n_use == 0:
        # Cannot happen while survivors exist, but stay safe
        selected = list(survivors)
        random.Random(seed).shuffle(selected)
        info.update({"mode": "empty_set_fallback", "n_returned": len(selected)})
        _report(info)
        return selected

    base = [(survivors[order[r]], counts_by_rank[r]) for r in range(n_use)]

    epoch_rng = random.Random(seed)
    selected = []
    total = 0
    n_epochs_started = 0
    while total < T:
        epoch = list(base)
        epoch_rng.shuffle(epoch)
        n_epochs_started += 1
        for i, c in epoch:
            selected.append(i)
            total += c
            if total >= T:
                break

    info.update({
        "mode": "repeat_H",
        "n_docs_used": n_use,
        "n_docs_extended_below_threshold": extended,
        "tokens_used_over_T": tok_use / float(T),
        "epochs": T / float(max(1, tok_use)),
        "epochs_started": n_epochs_started,
        "n_returned": len(selected),
        "tokens_exact": state["exact"],
        "seconds": time.time() - t0,
    })
    _report(info)
    return selected
