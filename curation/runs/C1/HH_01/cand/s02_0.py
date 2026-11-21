"""Filter-then-rank curator (Moore-Lewis cross-entropy difference).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
1. Drop junk: short docs, docs with too many non-text characters, docs with repeated lines.
2. Deduplicate on a hash of the normalized text (lowercased, whitespace collapsed), keeping the first copy.
3. Rank by Moore-Lewis cross-entropy difference under gpt2-token unigram LMs:
   score(doc) = mean over tokens t of [log p_target(t) - log p_pool(t)]
   (per-token mean, so long docs are not favored).
4. Walk the ranking, accumulating gpt2 token counts until T tokens; cycle the selection if short.
Tie-breaking is seeded by CURATE_SEED.
"""
import os
import random
import hashlib
import string
from collections import Counter

import numpy as np
import tiktoken

from common import T

MIN_CHARS = 200
MIN_CLEAN_FRAC = 0.70          # fraction of chars that must be ASCII letters/digits/space/punct
MIN_REPEAT_LINE_LEN = 10       # ignore very short repeated lines such as "}" or "-"
MAX_LM_TOKENS = 1000           # per-doc token cap for the unigram LMs
MAX_LM_CHARS = 8000            # char prefix encoded before the token cap (bounds runtime)
SMOOTH_K = 0.1                 # add-k smoothing constant
GPT2_VOCAB = 50257

_KEEP = set(string.ascii_letters + string.digits + " \t\r\n" + string.punctuation)
_STRIP_KEEP = {ord(c): None for c in _KEEP}  # translate() deletes kept chars, leaving the rest


def _is_junk(doc):
    if len(doc) < MIN_CHARS:
        return True
    unkept = len(doc.translate(_STRIP_KEEP))
    if (len(doc) - unkept) < MIN_CLEAN_FRAC * len(doc):
        return True
    counts = Counter(line.strip() for line in doc.splitlines())
    for line, c in counts.items():
        if c > 1 and len(line) >= MIN_REPEAT_LINE_LEN:
            return True
    return False


def _key(doc):
    norm = " ".join(doc.lower().split())
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()


def _lm_tokens(text, enc):
    if not isinstance(text, str) or not text:
        return np.zeros(0, dtype=np.int64)
    ids = enc.encode_ordinary(text[:MAX_LM_CHARS])[:MAX_LM_TOKENS]
    return np.asarray(ids, dtype=np.int64)


def _scores(pool_toks, tgt_toks):
    n = len(pool_toks)
    zeros = np.zeros(n)
    if n == 0 or not tgt_toks:
        return zeros
    tgt_all = np.concatenate(tgt_toks).astype(np.int64)
    pool_all = np.concatenate(pool_toks).astype(np.int64) if n else np.zeros(0, np.int64)
    if tgt_all.size == 0 or pool_all.size == 0:
        return zeros

    # Vocabulary size covers all seen ids; the extra +1 in V is the OOV bucket.
    M = max(GPT2_VOCAB, int(max(tgt_all.max(), pool_all.max())) + 1)
    ct = np.bincount(tgt_all, minlength=M).astype(np.float64)
    cp = np.bincount(pool_all, minlength=M).astype(np.float64)
    V = int(np.count_nonzero((ct > 0) | (cp > 0))) + 1  # union of seen tokens + OOV bucket
    k = SMOOTH_K

    # Ids never seen in a corpus get count 0, i.e. the smoothed OOV mass k / (N + kV).
    logpt = np.log((ct + k) / (ct.sum() + k * V))
    logpp = np.log((cp + k) / (cp.sum() + k * V))
    diff = logpt - logpp

    scores = np.zeros(n)
    for j, t in enumerate(pool_toks):
        if t.size:
            scores[j] = float(diff[t].mean())
    return scores


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    # 1 + 2: junk filter and deduplication (first copy wins)
    seen = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or _is_junk(d):
            continue
        k = _key(d)
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    if not keep:
        keep = list(range(n))  # fallback so we never return an empty curriculum

    enc = tiktoken.get_encoding("gpt2")

    # 3: Moore-Lewis cross-entropy difference ranking, seeded tie-breaking
    pool_toks = [_lm_tokens(docs[i], enc) for i in keep]
    tgt_toks = [
        _lm_tokens(t, enc)
        for t in target_sample
        if isinstance(t, str) and t.strip()
    ]
    tgt_toks = [t for t in tgt_toks if t.size]
    scores = _scores(pool_toks, tgt_toks)
    pos = list(range(len(keep)))
    rng.shuffle(pos)                      # random order among equal scores
    pos.sort(key=lambda p: -scores[p])    # stable sort keeps the shuffled order for ties
    ranked = [keep[p] for p in pos]

    # 4: accumulate gpt2 tokens in rank order until T
    selected = []
    counts = []
    total = 0
    for i in ranked:
        c = len(enc.encode_ordinary(docs[i]))
        selected.append(i)
        counts.append(c)
        total += c
        if total >= T:
            break

    # If still short, cycle the selected list until T tokens
    if total < T and selected and sum(counts) > 0:
        base, base_counts = list(selected), list(counts)
        k = 0
        while total < T:
            j = k % len(base)
            selected.append(base[j])
            total += base_counts[j]
            k += 1

    return selected
