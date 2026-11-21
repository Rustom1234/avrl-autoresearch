"""Curator: rank pool documents by vocabulary overlap and word-order agreement
with the target sample, behind a hard label-free 'wikitext format' gate.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Count the tokens of every pool document with tiktoken ("gpt2").
  2. Score every document by its IDF-weighted word overlap with the vocabulary
     of target_sample (share of the document's word mass that is made of words
     the target domain also uses).
  3. Score every document by word order: the share of its adjacent
     whitespace-token bigrams that also occur in target_sample
     (bigram_hit = hits / (bigrams + 20)).  Word-shuffled articles keep their
     vocabulary but lose their bigrams, so they score low here.
  4. Calibrate a bigram_hit threshold without labels: score the real target
     documents and word-shuffled copies of them (leave-one-out bigram set) and
     take the midpoint between the 5th percentile of the real scores and the
     95th percentile of the shuffled scores.
  5. Mark as unusable: documents below the threshold, documents where more
     than 3% of adjacent tokens are punctuation pairs (', ,', '. ,', ...),
     exact duplicates and empty documents.
  6. Format gate (calibrated on target_sample only):
       spaced_punct = share of whitespace tokens that are exactly one of
                      , . @-@ @,@ @.@ ( ) ; : " 's
       glued_punct  = share of whitespace tokens longer than one character
                      that end with ',' or '.'
       lo = 0.6 * p5(spaced_punct of target), hi = max(0.02, 2 * p95(glued_punct
       of target)); format_ok = spaced_punct >= lo and glued_punct <= hi.
  7. Partial-damage check: bigram_hit on consecutive windows of 200 whitespace
     tokens (windows shorter than 100 tokens are ignored, same +20 shrink); the
     minimum window score must be >= 0.8 * the calibrated threshold.
  8. Title dedupe: documents whose first line is an article title '= ... ='
     are grouped by the lowercased first line; only the one with the highest
     whole-document bigram_hit is kept (ties: the longest).  The winner is
     chosen among the documents that pass all the other gates, so an article
     is not lost because its best-scoring copy fails a different filter.
  9. Rank the gated documents by overlap score * bigram_hit, pick the best
     until the training budget of T tokens is filled, then shuffle the picked
     set with the seed from CURATE_SEED.  If the gated documents hold fewer
     than T tokens, only they are returned and the trainer cycles the list;
     documents that fail the gate are never added.
     Safety fallback only: if the gated set holds fewer than 0.25*T tokens the
     format gate (step 6) is dropped.
 10. The token total of the gated set as a fraction of T is logged to stderr.
"""
import os
import random
import re
import sys
from collections import Counter

import numpy as np

from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")

_BIGRAM_SHRINK = 20.0
_PUNCT_TOKENS = frozenset([",", ".", ";", ":", "!", "?"])
_PUNCT_PAIR_MAX = 0.03

# Format gate.
_FORMAT_TOKENS = frozenset(
    [",", ".", "@-@", "@,@", "@.@", "(", ")", ";", ":", '"', "'s"]
)
_SPACED_LO_FACTOR = 0.6
_GLUED_HI_FACTOR = 2.0
_GLUED_HI_FLOOR = 0.02

# Partial-damage (window) check.
_WINDOW = 200
_WINDOW_MIN = 100
_WINDOW_FACTOR = 0.8

# Fallback: minimum share of T the gated set must hold to keep the format gate.
_MIN_GATED_FRACTION = 0.25

# Article title line: '= Title =' but not a section heading '= = Section = ='.
_TITLE_RE = re.compile(r"^=\s+[^=\s].*?\s+=$|^=\s+[^=\s]\s+=$")


def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _token_counts(docs):
    """Number of gpt2 tokens in each document."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        counts = np.empty(len(docs), dtype=np.int64)
        chunk = 2000
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            try:
                encoded = enc.encode_ordinary_batch(batch, num_threads=8)
                lens = [len(e) for e in encoded]
            except Exception:
                lens = []
                for d in batch:
                    try:
                        lens.append(len(enc.encode_ordinary(d)))
                    except Exception:
                        lens.append(max(1, len(d) // 4))
            counts[start:start + len(batch)] = lens
        return counts
    except Exception:
        return np.array([max(1, len(d) // 4) for d in docs], dtype=np.int64)


def _overlap_scores(docs, target_sample):
    """IDF-weighted share of each document's words found in the target vocabulary."""
    from sklearn.feature_extraction.text import CountVectorizer

    n = len(docs)
    vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN, dtype=np.float32)
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:
        return np.zeros(n, dtype=np.float64)

    # Target vocabulary.
    word_re = re.compile(_TOKEN_PATTERN)
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(word_re.findall(t.lower()))

    vocab = vec.vocabulary_
    mask = np.zeros(X.shape[1], dtype=np.float64)
    for w in target_vocab:
        j = vocab.get(w)
        if j is not None:
            mask[j] = 1.0

    # Document frequency -> idf, so rare words (junk, other domains) weigh more.
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    # Sublinear term frequency: repeated words do not dominate.
    X.data = np.log1p(X.data)

    num = np.asarray(X @ (idf * mask)).ravel()
    den = np.asarray(X @ idf).ravel()
    # Small shrinkage so very short documents cannot get a perfect score.
    return num / (den + 30.0)


def _ws_tokens(text):
    """Lowercased whitespace tokens; punctuation tokens (',', '.', '@-@') are kept."""
    return text.lower().split()


def _shuffled_tokens(text, rng):
    """Tokens of the document with everything after the title line shuffled."""
    stripped = text.lstrip()
    head, sep, body = stripped.partition("\n")
    if sep:
        head_toks = _ws_tokens(head)
        body_toks = _ws_tokens(body)
    else:
        head_toks = []
        body_toks = _ws_tokens(stripped)
    rng.shuffle(body_toks)
    return head_toks + body_toks


def _calibrate(target_sample, seed):
    """Return (target bigram set, bigram_hit threshold).

    Real and word-shuffled target documents are both scored against a
    leave-one-out bigram set (bigrams seen in at least one *other* target
    document).  The threshold is the midpoint between the 5th percentile of the
    real scores and the 95th percentile of the shuffled scores.
    """
    tok_lists = [_ws_tokens(t) for t in target_sample]
    doc_sets = [set(zip(toks, toks[1:])) for toks in tok_lists]

    doc_count = Counter()
    for s in doc_sets:
        doc_count.update(s)
    bigram_set = set(doc_count)

    rng = random.Random(seed)
    real_scores = []
    shuf_scores = []
    for text, toks, own in zip(target_sample, tok_lists, doc_sets):
        nb = len(toks) - 1
        if nb <= 0:
            continue
        # Real document: every bigram is in its own set, so it must occur in
        # at least two target documents to be in the leave-one-out set.
        hits = 0
        for b in zip(toks, toks[1:]):
            if doc_count[b] >= 2:
                hits += 1
        real_scores.append(hits / (nb + _BIGRAM_SHRINK))

        stoks = _shuffled_tokens(text, rng)
        snb = len(stoks) - 1
        if snb <= 0:
            continue
        hits = 0
        for b in zip(stoks, stoks[1:]):
            c = doc_count.get(b, 0)
            if c and (c - (1 if b in own else 0)) >= 1:
                hits += 1
        shuf_scores.append(hits / (snb + _BIGRAM_SHRINK))

    if not real_scores or not shuf_scores:
        return bigram_set, 0.0

    lo_real = float(np.percentile(np.asarray(real_scores), 5))
    hi_shuf = float(np.percentile(np.asarray(shuf_scores), 95))
    return bigram_set, 0.5 * (lo_real + hi_shuf)


def _order_stats(docs, bigram_set):
    """Per document: bigram_hit, punctuation-pair fraction and the minimum
    bigram_hit over consecutive 200-token windows (inf if no window >= 100)."""
    n = len(docs)
    bigram_hit = np.zeros(n, dtype=np.float64)
    punct_frac = np.zeros(n, dtype=np.float64)
    min_window = np.full(n, np.inf, dtype=np.float64)
    contains = bigram_set.__contains__
    is_punct = _PUNCT_TOKENS.__contains__
    for i, d in enumerate(docs):
        toks = d.lower().split()
        ntok = len(toks)
        nb = ntok - 1
        if nb <= 0:
            continue
        hit_flags = list(map(contains, zip(toks, toks[1:])))
        hits = sum(hit_flags)
        bigram_hit[i] = hits / (nb + _BIGRAM_SHRINK)

        if ntok >= _WINDOW_MIN:
            if ntok <= _WINDOW:
                # Single window == whole document.
                min_window[i] = bigram_hit[i]
            else:
                best = float("inf")
                for s in range(0, ntok, _WINDOW):
                    wlen = min(_WINDOW, ntok - s)
                    if wlen < _WINDOW_MIN:
                        continue
                    wnb = wlen - 1
                    wh = sum(hit_flags[s:s + wnb])
                    sc = wh / (wnb + _BIGRAM_SHRINK)
                    if sc < best:
                        best = sc
                min_window[i] = best

        flags = list(map(is_punct, toks))
        if any(flags):
            pairs = sum(1 for a, b in zip(flags, flags[1:]) if a and b)
            punct_frac[i] = pairs / nb
    return bigram_hit, punct_frac, min_window


def _format_stats(texts):
    """Per document: (spaced_punct, glued_punct, number of whitespace tokens)."""
    n = len(texts)
    spaced = np.zeros(n, dtype=np.float64)
    glued = np.zeros(n, dtype=np.float64)
    ntoks = np.zeros(n, dtype=np.int64)
    is_fmt = _FORMAT_TOKENS.__contains__
    for i, d in enumerate(texts):
        toks = d.split()
        nt = len(toks)
        if nt == 0:
            continue
        ntoks[i] = nt
        spaced[i] = sum(map(is_fmt, toks)) / nt
        g = 0
        for t in toks:
            if len(t) > 1:
                c = t[-1]
                if c == "," or c == ".":
                    g += 1
        glued[i] = g / nt
    return spaced, glued, ntoks


def _title_keys(docs):
    """Lowercased, whitespace-normalised title line for documents that start
    with an article title '= ... ='; None otherwise."""
    keys = [None] * len(docs)
    for i, d in enumerate(docs):
        head = d.lstrip().partition("\n")[0].strip()
        if len(head) < 5 or head[0] != "=" or head[-1] != "=":
            continue
        if _TITLE_RE.match(head):
            keys[i] = _WS.sub(" ", head).lower()
    return keys


def _title_unique(mask, keys, bigram_hit, counts):
    """Among documents in `mask`, keep one per title: highest whole-document
    bigram_hit, ties broken by length, then by first position."""
    best = {}
    for i in np.flatnonzero(mask):
        k = keys[i]
        if k is None:
            continue
        cand = (float(bigram_hit[i]), int(counts[i]), -int(i))
        cur = best.get(k)
        if cur is None or cand > cur:
            best[k] = cand
    out = mask.copy()
    for i in np.flatnonzero(mask):
        k = keys[i]
        if k is None:
            continue
        if best[k][2] != -int(i):
            out[i] = False
    return out


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    counts = _token_counts(docs)
    scores = _overlap_scores(docs, target_sample)

    # Word-order score and its label-free threshold.
    bigram_set, threshold = _calibrate(target_sample, seed)
    bigram_hit, punct_frac, min_window = _order_stats(docs, bigram_set)
    if bigram_set:
        order_ok = bigram_hit >= threshold
        window_ok = min_window >= _WINDOW_FACTOR * threshold
        final = scores * bigram_hit
    else:
        # No target bigrams at all: the word-order score carries no information.
        order_ok = np.ones(n, dtype=bool)
        window_ok = np.ones(n, dtype=bool)
        final = scores
    punct_ok = punct_frac <= _PUNCT_PAIR_MAX

    # Format gate, calibrated on the target sample only.
    t_spaced, t_glued, t_ntoks = _format_stats(target_sample)
    t_valid = t_ntoks > 0
    if t_valid.any():
        lo = _SPACED_LO_FACTOR * float(np.percentile(t_spaced[t_valid], 5))
        hi = max(
            _GLUED_HI_FLOOR,
            _GLUED_HI_FACTOR * float(np.percentile(t_glued[t_valid], 95)),
        )
        spaced, glued, _ = _format_stats(docs)
        format_ok = (spaced >= lo) & (glued <= hi)
    else:
        lo, hi = 0.0, 1.0
        format_ok = np.ones(n, dtype=bool)

    # Exact duplicates (after whitespace/case normalisation): keep the first copy.
    seen = set()
    unique = np.ones(n, dtype=bool)
    for i, d in enumerate(docs):
        key = hash(_WS.sub(" ", d).strip().lower())
        if key in seen:
            unique[i] = False
        else:
            seen.add(key)

    base = unique & (counts > 0)
    prev_usable = base & order_ok & punct_ok
    no_format = prev_usable & window_ok

    title_keys = _title_keys(docs)

    gated = no_format & format_ok
    gated = _title_unique(gated, title_keys, bigram_hit, counts)
    gated_tokens = int(counts[gated].sum())
    gated_frac = gated_tokens / float(T)

    _log(
        "format gate: lo=%.4f hi=%.4f threshold=%.4f | prev_usable=%d "
        "window_ok=%d format_ok=%d gated=%d docs"
        % (
            lo,
            hi,
            threshold,
            int(prev_usable.sum()),
            int(no_format.sum()),
            int((no_format & format_ok).sum()),
            int(gated.sum()),
        )
    )
    _log(
        "gated set tokens = %d = %.3f * T (%s)"
        % (
            gated_tokens,
            gated_frac,
            "list will be cycled" if gated_tokens < T else "no cycling",
        )
    )

    if gated_frac >= _MIN_GATED_FRACTION:
        usable = gated
    else:
        # Safety fallback: the format gate removed almost everything, so it is
        # dropped (window check and title dedupe are kept).
        usable = _title_unique(no_format, title_keys, bigram_hit, counts)
        _log(
            "FALLBACK: gated set below %.2f*T, format gate dropped; "
            "usable tokens = %.3f * T"
            % (_MIN_GATED_FRACTION, int(counts[usable].sum()) / float(T))
        )

    if not usable.any():
        # Last resort so the trainer never gets an empty list.
        _log("FALLBACK: no usable documents, relaxing filters")
        usable = prev_usable
        if not usable.any():
            usable = base & punct_ok
        if not usable.any():
            usable = base
        if not usable.any():
            return list(range(n))

    order = np.argsort(-final, kind="stable")

    # Fill to T tokens with usable documents only.  If they hold fewer than T
    # tokens the list is returned as is and the trainer cycles it.
    selected = []
    total = 0
    for i in order:
        if not usable[i]:
            continue
        selected.append(int(i))
        total += int(counts[i])
        if total >= T:
            break

    _log(
        "selected %d docs, %d tokens = %.3f * T"
        % (len(selected), total, total / float(T))
    )

    random.Random(seed).shuffle(selected)
    return selected
