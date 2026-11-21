"""Curator: bigram-overlap score, exact dedup, calibrated "clean in-domain" gate.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Score every pool doc: s = b + 0.25*u + 0.1*title (b = bigram overlap with the
     target sample, u = unigram overlap), after exact dedup and a minimum length.
  2. Calibrate tau = 0.7 * (10th percentile of leave-one-out b over the target docs).
     Docs with b >= tau are CLEAN, everything else is FILLER.
  3. If the CLEAN docs hold >= 1.02*T tokens: take them by s descending up to the
     budget and shuffle. Otherwise repeat the CLEAN docs (up to 3 passes), fill any
     remainder with the best FILLER docs having b >= 0.5*tau, then with one more
     partial pass of CLEAN docs.
  4. Fallback to the legacy behaviour (top-s docs up to 1.02*T, shuffled) if fewer
     than 200 docs are CLEAN or the time check exceeds 150 s.
"""
import os
import time
import random
import hashlib

from common import T

MAX_SCORE_TOKENS = 2000    # score each doc on its first N whitespace tokens
MIN_TOKENS = 50            # drop docs shorter than this (whitespace tokens)
BUDGET_FACTOR = 1.02       # keep about 1.02 * T GPT-2 tokens
TIME_BUDGET_EXACT = 110.0  # legacy path: after this, token counting uses len/4
TIME_LIMIT_GATE = 150.0    # new path: past this, fall back to legacy behaviour
BATCH = 512

TAU_FACTOR = 0.7           # tau = TAU_FACTOR * 10th percentile of leave-one-out b
TAU_PERCENTILE = 10.0
FILLER_TAU_FACTOR = 0.5    # filler must satisfy b >= FILLER_TAU_FACTOR * tau
MAX_PASSES = 3
MIN_CLEAN_DOCS = 200


def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")


def _percentile(values, pct):
    """Linear-interpolation percentile (same convention as numpy's default)."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return vals[0]
    pos = (pct / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _calibrate_tau(target_sample):
    """tau = TAU_FACTOR * 10th percentile of leave-one-out bigram overlap."""
    doc_tokens = [doc.split() for doc in target_sample]
    doc_bigram_sets = [set(zip(toks, toks[1:])) for toks in doc_tokens]

    df = {}
    for bg_set in doc_bigram_sets:
        for bg in bg_set:
            df[bg] = df.get(bg, 0) + 1

    loo = []
    for toks, own in zip(doc_tokens, doc_bigram_sets):
        head = toks[:MAX_SCORE_TOKENS]
        if len(head) < 2:
            continue
        hits = 0
        for bg in zip(head, head[1:]):
            need = 2 if bg in own else 1
            if df.get(bg, 0) >= need:
                hits += 1
        loo.append(hits / (len(head) - 1))

    p = _percentile(loo, TAU_PERCENTILE)
    if p is None:
        return None
    return TAU_FACTOR * p


def _get_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except Exception:
        return None


def _legacy_select(docs, scored, rng, enc, t_start, tok_cache):
    """Previous behaviour: top-s docs up to ~1.02*T tokens, then a seeded shuffle."""
    if not scored:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    ranked = [i for _, i in sorted(scored, key=lambda x: (-x[0], x[1]))]
    budget = BUDGET_FACTOR * T

    selected = []
    total = 0.0
    pos = 0
    done = False
    while pos < len(ranked) and not done:
        batch_idx = ranked[pos:pos + BATCH]
        pos += BATCH

        missing = [i for i in batch_idx if i not in tok_cache]
        if missing:
            lengths = None
            if enc is not None and (time.time() - t_start) < TIME_BUDGET_EXACT:
                try:
                    encoded = enc.encode_ordinary_batch([docs[i] for i in missing])
                    lengths = [len(e) for e in encoded]
                except Exception:
                    lengths = None
            if lengths is None:
                lengths = [len(docs[i]) / 4.0 for i in missing]
            local = dict(zip(missing, lengths))
        else:
            local = {}

        for i in batch_idx:
            n_tok = tok_cache[i] if i in tok_cache else local[i]
            selected.append(i)
            total += n_tok
            if total >= budget:
                done = True
                break

    rng.shuffle(selected)
    return selected


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    def timed_out():
        return (time.time() - t_start) > TIME_LIMIT_GATE

    # Target unigram vocabulary and adjacent-bigram set (case preserved).
    vocab = set()
    bigrams = set()
    for doc in target_sample:
        toks = doc.split()
        vocab.update(toks)
        bigrams.update(zip(toks, toks[1:]))

    tau = _calibrate_tau(target_sample)

    # Score, filter and deduplicate the pool.
    seen = set()
    scored = []   # (score, index)
    b_of = {}     # index -> bigram overlap
    for i, text in enumerate(docs):
        toks = text.split()
        if len(toks) < MIN_TOKENS:
            continue

        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        head = toks[:MAX_SCORE_TOKENS]
        n = len(head)
        u = sum(1 for t in head if t in vocab) / n
        if n > 1:
            b = sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1)
        else:
            b = 0.0

        score = b + 0.25 * u
        if _has_title_line(text):
            score += 0.1
        scored.append((score, i))
        b_of[i] = b
    del seen

    enc = _get_encoder()
    tok_cache = {}

    def fallback():
        return _legacy_select(docs, scored, rng, enc, t_start, tok_cache)

    if not scored or tau is None:
        return fallback()

    # Best first; index as a deterministic tie-break.
    scored.sort(key=lambda x: (-x[0], x[1]))

    clean = [i for _, i in scored if b_of[i] >= tau]          # s descending
    if len(clean) < MIN_CLEAN_DOCS or timed_out():
        return fallback()

    def count_tokens(indices):
        """Fill tok_cache for indices. Returns False if the time limit was hit."""
        pos = 0
        while pos < len(indices):
            if timed_out():
                return False
            batch_idx = [i for i in indices[pos:pos + BATCH] if i not in tok_cache]
            pos += BATCH
            if not batch_idx:
                continue
            lengths = None
            if enc is not None:
                try:
                    encoded = enc.encode_ordinary_batch([docs[i] for i in batch_idx])
                    lengths = [len(e) for e in encoded]
                except Exception:
                    lengths = None
            if lengths is None:
                lengths = [len(docs[i]) / 4.0 for i in batch_idx]
            for i, n_tok in zip(batch_idx, lengths):
                tok_cache[i] = n_tok
        return True

    if not count_tokens(clean):
        return fallback()

    budget = BUDGET_FACTOR * T
    C = float(sum(tok_cache[i] for i in clean))
    if C <= 0:
        return fallback()

    # Case A: enough clean text -- best clean docs up to the budget, shuffled.
    if C >= budget:
        selected = []
        total = 0.0
        for i in clean:
            selected.append(i)
            total += tok_cache[i]
            if total >= budget:
                break
        rng.shuffle(selected)
        return selected

    # Case B: not enough clean text -- repeat clean docs instead of adding junk.
    k = max(1, min(MAX_PASSES, int(budget // C)))
    remainder = budget - k * C

    filler_sel = []
    partial = []
    if remainder > 0:
        filler_floor = FILLER_TAU_FACTOR * tau
        filler = [i for _, i in scored
                  if b_of[i] < tau and b_of[i] >= filler_floor]  # s descending

        pos = 0
        while remainder > 0 and pos < len(filler):
            batch_idx = filler[pos:pos + BATCH]
            pos += BATCH
            if not count_tokens(batch_idx):
                return fallback()
            for i in batch_idx:
                filler_sel.append(i)
                remainder -= tok_cache[i]
                if remainder <= 0:
                    break

        # Not enough acceptable filler: one more partial pass of clean docs.
        if remainder > 0:
            for i in clean:
                partial.append(i)
                remainder -= tok_cache[i]
                if remainder <= 0:
                    break

    out = []
    for p in range(k):
        this_pass = list(clean)
        if p == 0 and filler_sel:
            this_pass.extend(filler_sel)   # filler is mixed into the first pass
        rng.shuffle(this_pass)
        out.extend(this_pass)
    if partial:
        rng.shuffle(partial)
        out.extend(partial)

    return out
