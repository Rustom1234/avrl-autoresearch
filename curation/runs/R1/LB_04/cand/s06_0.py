"""Curator: target-similarity filter + junk removal + content-level redundancy removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Char n-gram TF-IDF + logistic regression (target vs. random pool subsample),
     followed by one round of self-training on the pool.
  2. Drop junk (too short, low alphabetic fraction, low in-target-vocabulary fraction,
     and low word-order coherence compared to a random shuffle).
  3. Intra-doc repetition rule: drop docs whose distinct sentence units hold < 70% of
     their unit chars (>= 5 units), or whose zlib compression ratio is < 0.22.
  4. Cross-doc near-duplicate rule: in descending score order (ties by index), drop a
     doc if > 50% of its char-weighted sentence units were already seen in kept docs.
     Docs with < 3 units rely on the first-500-char hash key, which is also applied
     to every doc as an additional check.
  5. Keep the top-scoring documents until ~1.0 * T tokens.
  6. Seeded shuffle of the kept set.

Logging goes to stderr and to the module-level LAST_STATS dict (no stdout, no files).
"""
import os
import random
import re
import sys
import time
import zlib
import hashlib
from collections import Counter

import numpy as np

from common import T

HEAD_CHARS = 3000          # chars of each doc used for featurization
POOL_CHAR_BUDGET = 60_000_000  # deterministic cap on total chars vectorized (time safety)
NEG_SUBSAMPLE = 4000
MAX_SELF_POS = 6000        # caps on self-training set size (time safety)
MAX_SELF_NEG = 12000
TOKEN_BUDGET = 1.0 * T

UNIT_TEXT_CHARS = 40000    # chars of each doc used for redundancy analysis
MIN_UNIT_CHARS = 40
MIN_UNITS_REPETITION = 5
MIN_DISTINCT_FRAC = 0.70
MIN_ZLIB_RATIO = 0.22
MIN_UNITS_NEARDUP = 3
MAX_SEEN_FRAC = 0.5
TIME_GUARD_SECONDS = 150.0  # after this, the leftover token total is estimated

_WORD_RE = re.compile(r"[^\W\d_]+")
_ALPHA_SPACE_RE = re.compile(r"[^\W\d_]|\s")
_WS_RE = re.compile(r"\s+")
_ORDER_RE = re.compile(r"[^\W\d_]+|[^\w\s]")
_UNIT_SPLIT_RE = re.compile(r"\n+| \. |(?<=[.!?])\s+")

LAST_STATS = {}


def _seed():
    try:
        return int(os.environ.get("CURATE_SEED", 0))
    except ValueError:
        return 0


def _log(msg):
    try:
        sys.stderr.write("[curate] " + msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _score(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    rng = random.Random(seed)

    cap = int(max(600, min(HEAD_CHARS, POOL_CHAR_BUDGET // max(n, 1))))

    tgt_heads = [t[:HEAD_CHARS] for t in target_sample]
    sub_idx = rng.sample(range(n), min(NEG_SUBSAMPLE, n))
    sub_heads = [docs[i][:HEAD_CHARS] for i in sub_idx]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )
    X_fit = vec.fit_transform(tgt_heads + sub_heads)
    y_fit = np.array([1] * len(tgt_heads) + [0] * len(sub_heads))
    X_tgt = X_fit[: len(tgt_heads)]

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf.fit(X_fit, y_fit)

    X_pool = vec.transform([d[:cap] for d in docs])
    scores = clf.predict_proba(X_pool)[:, 1]

    try:
        from scipy.sparse import vstack

        order = np.argsort(-scores, kind="stable")
        n_pos = int(0.15 * n)
        n_neg = int(0.50 * n)
        if n_pos >= 10 and n_neg >= 10:
            pos = order[:n_pos].tolist()
            neg = order[n - n_neg:].tolist()
            if len(pos) > MAX_SELF_POS:
                pos = rng.sample(pos, MAX_SELF_POS)
            if len(neg) > MAX_SELF_NEG:
                neg = rng.sample(neg, MAX_SELF_NEG)
            X2 = vstack([X_tgt, X_pool[pos], X_pool[neg]]).tocsr()
            y2 = np.array([1] * (X_tgt.shape[0] + len(pos)) + [0] * len(neg))
            clf2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
            clf2.fit(X2, y2)
            scores = clf2.predict_proba(X_pool)[:, 1]
    except Exception:
        pass

    return np.asarray(scores, dtype=np.float64)


def _target_vocab(target_sample):
    cnt = Counter()
    for t in target_sample:
        cnt.update(_WORD_RE.findall(t.lower()))
    return {w for w, c in cnt.items() if c >= 2}


def _junk_reason(doc, vocab, bigrams_set, seed_val, doc_idx):
    """Return None if the doc passes the junk rules, else the name of the rule
    that drops it ("short_alpha", "vocab" or "order"). Same rules as before,
    evaluated once per doc."""
    if len(doc.strip()) < 200:
        return "short_alpha"

    lines = doc.splitlines()
    if len(lines) > 1:
        lines = lines[1:]
    body = "".join(lines)[:40000]
    if not body:
        return "short_alpha"

    if len(_ALPHA_SPACE_RE.findall(body)) / len(body) < 0.7:
        return "short_alpha"

    body_lower = body.lower()
    if vocab:
        words = _WORD_RE.findall(body_lower)
        if not words:
            return "vocab"
        in_vocab = sum(1 for w in words if w in vocab)
        if in_vocab / len(words) < 0.5:
            return "vocab"

    tokens = _ORDER_RE.findall(body_lower)
    if len(tokens) < 30:
        return None

    nt = len(tokens)
    hits = [(tokens[i], tokens[i + 1]) in bigrams_set for i in range(nt - 1)]
    h_real = sum(hits) / (nt - 1)

    rng = random.Random(seed_val + doc_idx)
    shuffled = tokens[:]
    rng.shuffle(shuffled)
    h_shuf = 0
    for i in range(nt - 1):
        if (shuffled[i], shuffled[i + 1]) in bigrams_set:
            h_shuf += 1
    h_shuf /= (nt - 1)

    thresh = 1.6 * h_shuf + 0.03
    if h_real < thresh:
        return "order"

    # Window test
    window_count = 0
    low_window_count = 0
    for i in range(0, nt, 150):
        wlen = min(150, nt - i)
        if wlen < 60:
            break
        window_count += 1
        h_real_w = sum(hits[i:i + wlen - 1]) / (wlen - 1)
        if h_real_w < thresh:
            low_window_count += 1

    if window_count > 0 and (low_window_count / window_count) > 0.25:
        return "order"

    return None


def _dedup_key(doc):
    norm = _WS_RE.sub(" ", doc[:2000].lower()).strip()[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _units(text):
    """Sentence units of `text` as parallel lists (hashes, char lengths)."""
    hashes = []
    lens = []
    for u in _UNIT_SPLIT_RE.split(text.lower()):
        if len(u) < MIN_UNIT_CHARS:
            continue
        u = " ".join(u.split())
        if len(u) < MIN_UNIT_CHARS:
            continue
        hashes.append(hash(u))
        lens.append(len(u))
    return hashes, lens


def _is_repetitive(text, hashes, lens):
    if len(hashes) >= MIN_UNITS_REPETITION:
        total = sum(lens)
        if total > 0:
            seen_local = set()
            distinct = 0
            for h, l in zip(hashes, lens):
                if h not in seen_local:
                    seen_local.add(h)
                    distinct += l
            if distinct / total < MIN_DISTINCT_FRAC:
                return True
    raw = text.encode("utf-8", "ignore")
    if raw and len(zlib.compress(raw)) / len(raw) < MIN_ZLIB_RATIO:
        return True
    return False


def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts)]
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count


def select(docs, target_sample):
    t_start = time.monotonic()
    seed = _seed()
    n = len(docs)
    if n == 0:
        return []

    try:
        scores = _score(docs, target_sample, seed)
    except Exception:
        scores = np.zeros(n, dtype=np.float64)

    vocab = _target_vocab(target_sample)
    bigrams_set = set()
    for t in target_sample:
        tokens = _ORDER_RE.findall(t.lower())
        for i in range(len(tokens) - 1):
            bigrams_set.add((tokens[i], tokens[i + 1]))

    rule_drops = {
        "short_alpha": 0,
        "vocab": 0,
        "order": 0,
        "repetition": 0,
        "near_dup": 0,
        "dedup": 0,
    }

    # Pass 1: junk rules + intra-doc repetition rule (order independent).
    survivors = []
    unit_data = {}
    for i, d in enumerate(docs):
        reason = _junk_reason(d, vocab, bigrams_set, seed, i)
        if reason is not None:
            rule_drops[reason] += 1
            continue
        text = d[:UNIT_TEXT_CHARS]
        hashes, lens = _units(text)
        if _is_repetitive(text, hashes, lens):
            rule_drops["repetition"] += 1
            continue
        survivors.append(i)
        unit_data[i] = (hashes, lens)

    if not survivors:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    # Pass 2: cross-doc near-duplicate rule, best score first, ties by doc index.
    cand = np.asarray(survivors, dtype=np.int64)  # ascending index
    cand = cand[np.argsort(-scores[cand], kind="stable")]

    seen_units = set()
    seen_keys = set()
    clean = []
    for i in cand.tolist():
        hashes, lens = unit_data[i]
        if len(hashes) >= MIN_UNITS_NEARDUP:
            total = sum(lens)
            dup = 0
            for h, l in zip(hashes, lens):
                if h in seen_units:
                    dup += l
            if total > 0 and dup / total > MAX_SEEN_FRAC:
                rule_drops["near_dup"] += 1
                continue
        k = _dedup_key(docs[i])
        if k in seen_keys:
            rule_drops["dedup"] += 1
            continue
        seen_keys.add(k)
        seen_units.update(hashes)
        clean.append(i)
    del unit_data, seen_units, seen_keys

    if not clean:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    # Fill to the token budget by score; keep counting for the clean-total log.
    count_fn = _token_counter()
    kept = []
    total = 0            # tokens in the kept (selected) set
    clean_tokens = 0     # tokens in all clean docs
    counted_chars = 0
    estimated = False
    filled = False
    chunk = 2000
    for s in range(0, len(clean), chunk):
        part = clean[s:s + chunk]
        if filled and time.monotonic() - t_start > TIME_GUARD_SECONDS:
            # Out of time: estimate the remaining clean tokens from chars.
            rest_chars = sum(len(docs[i]) for i in clean[s:])
            ratio = clean_tokens / counted_chars if counted_chars else 0.25
            clean_tokens += int(rest_chars * ratio)
            estimated = True
            break
        counts = count_fn([docs[i] for i in part])
        for i, c in zip(part, counts):
            clean_tokens += c
            counted_chars += len(docs[i])
            if not filled:
                kept.append(i)
                total += c
                if total >= TOKEN_BUDGET:
                    filled = True

    random.Random(seed).shuffle(kept)

    stats = {
        "n_docs": n,
        "drops": dict(rule_drops),
        "n_clean_docs": len(clean),
        "clean_tokens": int(clean_tokens),
        "clean_tokens_over_T": clean_tokens / float(T),
        "clean_tokens_estimated": estimated,
        "n_kept_docs": len(kept),
        "kept_tokens": int(total),
        "kept_tokens_over_T": total / float(T),
        "min_kept_score": float(min(scores[i] for i in kept)) if kept else None,
        "seconds": time.monotonic() - t_start,
    }
    LAST_STATS.clear()
    LAST_STATS.update(stats)

    _log("pool docs: %d" % n)
    _log("junk drops: short_alpha=%d vocab=%d order=%d"
         % (rule_drops["short_alpha"], rule_drops["vocab"], rule_drops["order"]))
    _log("repetition rule removed: %d docs" % rule_drops["repetition"])
    _log("near-duplicate rule removed: %d docs (+%d by first-500-char key)"
         % (rule_drops["near_dup"], rule_drops["dedup"]))
    _log("clean docs: %d, clean tokens: %d = %.3f * T%s"
         % (len(clean), clean_tokens, clean_tokens / float(T),
            " (partly estimated)" if estimated else ""))
    _log("kept docs: %d, kept tokens: %d = %.3f * T, elapsed %.1fs"
         % (len(kept), total, total / float(T), stats["seconds"]))

    return kept
