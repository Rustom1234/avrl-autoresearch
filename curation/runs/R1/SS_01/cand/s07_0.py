"""Curator: strict target-likeness filter (tier A) plus a rescue tier B of new
in-domain articles, with an explicit training sequence that ends on pure tier A.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Tier A: first-occurrence dedupe, junk filter, bigram hit rate on the first 2000
        word tokens, threshold calibrated leave-one-out on the target sample.
Tier B: only when tier A is under T tokens. Titled articles not in tier A whose
        title and prefix are new, with a slightly relaxed bigram floor and a
        unigram-vocabulary check against character noise.
Sequence: [early block: tier B + extra tier A, shuffled] + [final block: one
        shuffled pass of tier A]. Total is between T and 1.02*T tokens.
"""
import os
import random
import re
import string
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SCORE_TOKENS = 2000      # word tokens per doc used for scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02         # upper bound on sequence size, as a multiple of T
TARGET_FILL = 1.01       # aimed sequence size, as a multiple of T

B_SCORE_SLACK = 0.08     # tier B bigram floor = tier A threshold - this
B_MAX_FRAC = 0.35        # tier B is at most this fraction of T
B_VOCAB_PCTL = 2.0
B_VOCAB_MARGIN = 0.05
FINAL_MAX_FRAC = 0.60    # final pure tier A block is at most this fraction of T

TIKTOKEN_DEADLINE = 100  # seconds since start: stop exact tokenising of tier A
B_SCAN_DEADLINE = 120    # seconds since start: stop scanning tier B candidates
B_TOKEN_DEADLINE = 135   # seconds since start: stop exact tokenising of tier B

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}

_TITLE_RE = re.compile(r"^=\s+([^=].*?)\s+=$|^=\s+([^=\s])\s+=$")
_NON_LETTER_RE = re.compile(r"[^a-z]+")

_ENC = None
_ENC_FAILED = False


def _head_tokens(text, n=SCORE_TOKENS):
    """First n lowercase whitespace tokens."""
    return [t.lower() for t in text.split(None, n)[:n]]


def _clean_frac(text):
    if not text:
        return 0.0
    bad = len(text.translate(_DELETE_ALLOWED))
    return 1.0 - bad / len(text)


def _hit_rate(toks, contains):
    n = len(toks) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(toks, toks[1:]):
        if contains(b):
            hits += 1
    return hits / n


def _vocab_frac(toks, contains):
    """Fraction of alphabetic tokens for which contains(token) is true."""
    n = 0
    hits = 0
    for t in toks:
        if t.isalpha():
            n += 1
            if contains(t):
                hits += 1
    return hits / n if n else 0.0


def _title_key(text):
    """Title key if the first line looks like '= title =', else None."""
    stripped = text.lstrip()
    if not stripped.startswith("="):
        return None
    line = stripped.split("\n", 1)[0].strip()
    m = _TITLE_RE.match(line)
    if not m:
        return None
    title = (m.group(1) or m.group(2) or "").strip()
    if not title or title.startswith("=") or title.endswith("="):
        return None  # section heading such as '= = History = ='
    low = title.lower()
    key = _NON_LETTER_RE.sub("", low)
    if not key:
        # titles with no letters (e.g. years): keep them distinct from each other
        key = "#" + "".join(low.split())
    return key


def _get_encoder():
    global _ENC, _ENC_FAILED
    if _ENC is None and not _ENC_FAILED:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("gpt2")
        except Exception:
            _ENC_FAILED = True
    return _ENC


def _count_tokens(texts, t0, deadline=TIKTOKEN_DEADLINE):
    """gpt2 token counts; falls back to len/4 if tiktoken is unavailable or slow."""
    counts = [max(1, len(t) // 4) for t in texts]
    enc = _get_encoder()
    if enc is None:
        return counts
    try:
        chunk = 512
        for s in range(0, len(texts), chunk):
            if time.time() - t0 > deadline:
                break
            batch = enc.encode_ordinary_batch(texts[s:s + chunk])
            for j, ids in enumerate(batch):
                counts[s + j] = max(1, len(ids))
    except Exception:
        pass
    return counts


def _report(tok_a, tok_b, n_b, passes_started, passes_equiv):
    print(
        "curate: tokens(A)/T=%.4f tokens(B)/T=%.4f tierB_docs=%d "
        "tierA_passes=%d (%.2f pass-equivalents)"
        % (tok_a / T, tok_b / T, n_b, passes_started, passes_equiv),
        flush=True,
    )


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- reference bigram set + leave-one-out calibration --------------------
    df = Counter()            # number of target docs containing each bigram
    udf = Counter()           # number of target docs containing each alphabetic token
    target_heads = []
    for t in target_sample:
        toks = t.lower().split()
        df.update(set(zip(toks, toks[1:])))
        udf.update(set(w for w in toks if w.isalpha()))
        target_heads.append(toks[:SCORE_TOKENS])
    B = set(df)
    V = set(udf)

    loo = []
    loo_vocab = []
    for toks in target_heads:
        if len(toks) < 2:
            continue
        # bigram is in the other 199 docs iff its doc-frequency is >= 2
        loo.append(_hit_rate(toks, lambda b: df.get(b, 0) >= 2))
        loo_vocab.append(_vocab_frac(toks, lambda w: udf.get(w, 0) >= 2))
    threshold = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    threshold = max(threshold, 0.0)
    vocab_threshold = (
        float(np.percentile(loo_vocab, B_VOCAB_PCTL)) - B_VOCAB_MARGIN
        if loo_vocab else 0.0
    )

    # ---- dedupe, score, junk filter (tier A pipeline) ------------------------
    n_docs = len(docs)
    prefix_hash = [None] * n_docs   # 200-char normalised prefix hash per valid doc
    seen_full = set()
    seen_prefix = set()
    cand = []                 # (score, idx) for unique, non-junk docs
    b_contains = B.__contains__
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        norm = " ".join(text.lower().split())
        if not norm:
            continue
        h = hash(norm)
        p = hash(norm[:PREFIX_CHARS])
        prefix_hash[i] = p
        if h in seen_full or p in seen_prefix:
            continue
        seen_full.add(h)
        seen_prefix.add(p)

        toks = _head_tokens(text)
        if len(toks) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue
        cand.append((_hit_rate(toks, b_contains), i))
    del seen_full, seen_prefix

    if not cand:
        idx = list(range(n_docs))
        rng.shuffle(idx)
        _report(0, 0, 0, 0, 0.0)
        return idx

    # best first, deterministic tie-break on index
    cand.sort(key=lambda x: (-x[0], x[1]))
    kept = [i for s, i in cand if s >= threshold]

    budget = T * OVERSHOOT

    if not kept:
        # Nothing passed the threshold: fall back to the best-scoring clean docs.
        total = 0
        for s, i in cand:
            if total >= budget:
                break
            kept.append(i)
            total += max(1, len(docs[i]) // 4)
        rng.shuffle(kept)
        _report(0, 0, 0, 0, 0.0)
        return kept

    counts = _count_tokens([docs[i] for i in kept], t0, TIKTOKEN_DEADLINE)
    tok_a = sum(counts)

    # ---- tier A alone is enough: behave exactly as before --------------------
    if tok_a >= T:
        total = 0
        out = []
        for i, c in zip(kept, counts):
            if total >= budget:
                break
            out.append(i)
            total += c
        rng.shuffle(out)
        _report(tok_a, 0, 0, 1, min(total, tok_a) / tok_a)
        return out

    cnt = dict(zip(kept, counts))
    a_set = set(kept)

    # ---- tier B: rescue new in-domain articles -------------------------------
    a_prefixes = set(prefix_hash[i] for i in kept)
    a_keys = set()
    for i in kept:
        k = _title_key(docs[i])
        if k is not None:
            a_keys.add(k)

    score_of = {i: s for s, i in cand}
    b_floor = threshold - B_SCORE_SLACK
    v_contains = V.__contains__
    best = {}                 # title key -> (score, idx)
    for i, text in enumerate(docs):
        if i in a_set:
            continue
        p = prefix_hash[i]
        if p is None or p in a_prefixes:
            continue
        if (i & 255) == 0 and time.time() - t0 > B_SCAN_DEADLINE:
            break
        key = _title_key(text)
        if key is None or key in a_keys:
            continue
        toks = _head_tokens(text)
        if len(toks) < MIN_TOKENS:
            continue
        s = score_of.get(i)
        if s is None:
            if _clean_frac(text) < MIN_CLEAN_FRAC:
                continue
            s = _hit_rate(toks, b_contains)
        if s < b_floor:
            continue
        if _vocab_frac(toks, v_contains) < vocab_threshold:
            continue
        cur = best.get(key)
        if cur is None or s > cur[0]:   # ties keep the earliest index
            best[key] = (s, i)

    b_cands = sorted(best.values(), key=lambda x: (-x[0], x[1]))
    b_budget = min(T - tok_a, B_MAX_FRAC * T)
    tier_b = []
    tok_b = 0
    chunk = 256
    done = False
    for s0 in range(0, len(b_cands), chunk):
        if done:
            break
        part = [i for _, i in b_cands[s0:s0 + chunk]]
        part_counts = _count_tokens([docs[i] for i in part], t0, B_TOKEN_DEADLINE)
        for i, c in zip(part, part_counts):
            remaining = b_budget - tok_b
            if remaining < 32:
                done = True
                break
            if c <= remaining:
                tier_b.append(i)
                cnt[i] = c
                tok_b += c

    # ---- sequence -------------------------------------------------------------
    # Final block: one pass of tier A, truncated to best-scoring docs if large.
    final_cap = FINAL_MAX_FRAC * T
    if tok_a > final_cap:
        final = []
        final_tok = 0
        for i in kept:        # kept is sorted by score, best first
            c = cnt[i]
            if final_tok + c > final_cap:
                break
            final.append(i)
            final_tok += c
        if not final:
            final = [kept[0]]
            final_tok = cnt[kept[0]]
        leftover = kept[len(final):]
    else:
        final = list(kept)
        final_tok = tok_a
        leftover = []

    target = TARGET_FILL * T
    hi = OVERSHOOT * T
    total = final_tok + tok_b
    early_a = []
    a_used = final_tok
    passes_started = 1

    def _add(order, total, a_used, upper):
        for i in order:
            if total >= target:
                break
            c = cnt[i]
            if total + c > upper:
                continue
            early_a.append(i)
            total += c
            a_used += c
        return total, a_used

    # tier A docs that were cut from the final block come first: they are new text
    if leftover and total < target:
        order = list(leftover)
        rng.shuffle(order)
        total, a_used = _add(order, total, a_used, hi)

    # additional shuffled passes of tier A only as needed
    while total < target:
        order = list(kept)
        rng.shuffle(order)
        before = total
        total, a_used = _add(order, total, a_used, hi)
        if total == before:
            break
        passes_started += 1

    # safety: never return fewer than T tokens (would make the trainer wrap)
    if total < T:
        passes_started += 1
        while total < T:
            order = list(kept)
            rng.shuffle(order)
            for i in order:
                if total >= T:
                    break
                early_a.append(i)
                total += cnt[i]
                a_used += cnt[i]

    early = list(tier_b) + early_a
    rng.shuffle(early)
    rng.shuffle(final)

    _report(tok_a, tok_b, len(tier_b), passes_started, a_used / tok_a)
    print(
        "curate: sequence tokens/T=%.4f early_docs=%d final_docs=%d elapsed=%.1fs"
        % (total / T, len(early), len(final), time.time() - t0),
        flush=True,
    )

    # no global shuffle: the end of training is the pure tier A block
    return early + final
