"""Curator: greedy selection by lexical overlap with the target sample, exact token
budget, and a coarse easy-to-target curriculum.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Use tiktoken on a seeded sample of the pool to get tokens per character (used only
   as a fallback estimate of document length).
2. Build word-frequency distributions for the target sample and for the pool
   (background). Each word gets a weight log(mix(p_target, p_pool) / p_pool).
3. Score each pool document by its mean word weight, drop exact duplicates and
   obvious junk.
4. Walk the documents best-first, counting real gpt2 tokens (batched, 8 threads),
   and stop at the document that crosses T tokens (no margin).
5. Order the chosen documents in three blocks by score rank, split by cumulative
   tokens: lowest-scoring 30% first, middle 40% next, highest-scoring 30% last.
   Shuffle within each block (seeded).
"""
import os
import re
import math
import time
import random
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_WS = re.compile(r"\s+")

SCORE_CHARS = 4000      # characters of each doc used for scoring
TOKEN_SAMPLE = 600      # docs tokenized to estimate token statistics
BG_SAMPLE = 4000        # docs used for the background word distribution
LAMBDA = 0.7            # weight of target distribution in the mixture
MAX_W = 4.0             # cap on a single word's weight
TOK_CHUNK = 2000        # docs per tokenizer batch
TOK_THREADS = 8         # tokenizer threads
TIME_GUARD = 100.0      # seconds after which token counts are estimated
HIGH_FRAC = 0.30        # highest-scoring share of tokens, trained last
LOW_FRAC = 0.30         # lowest-scoring share of tokens, trained first


def _words(text):
    return _WORD.findall(text[:SCORE_CHARS].lower())


def _get_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except Exception:
        return None


def _token_stats(docs, rng, enc):
    """Return (average tokens per doc, tokens per character) from a sample."""
    n = len(docs)
    sample = rng.sample(range(n), min(TOKEN_SAMPLE, n))
    tpc = 0.25
    if enc is not None:
        try:
            tok = 0
            chars = 0
            for i in sample:
                d = docs[i]
                if not d:
                    continue
                tok += len(enc.encode(d, disallowed_special=()))
                chars += len(d)
            if chars > 0 and tok > 0:
                tpc = tok / chars
        except Exception:
            pass
    total_chars = sum(len(d) for d in docs)
    avg_tokens = tpc * total_chars / max(1, n)
    return avg_tokens, tpc


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    enc = _get_encoder()

    # 1. Token statistics
    avg_tokens, tpc = _token_stats(docs, rng, enc)
    avg_tokens = max(avg_tokens, 1.0)

    # 2. Word frequencies
    tgt = Counter()
    for t in target_sample:
        tgt.update(_words(t))
    n_t = sum(tgt.values())

    bg = Counter()
    for i in rng.sample(range(n), min(BG_SAMPLE, n)):
        bg.update(_words(docs[i]))
    n_b = sum(bg.values())

    if n_t == 0 or n_b == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    floor_b = 0.5 / n_b
    unk_w = math.log(1.0 - LAMBDA)   # word seen in neither target nor background
    weights = {}
    for w, c in bg.items():
        pb = c / n_b
        pt = tgt.get(w, 0) / n_t
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt.items():
        if w not in weights:
            pt = c / n_t
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    # 3. Score documents, drop duplicates and junk
    wget = weights.get
    seen = set()
    scored = []
    rest = []   # low-priority fallback (duplicates excluded)
    for i, d in enumerate(docs):
        if not d:
            continue
        key = hash(_WS.sub(" ", d.strip().lower()))
        if key in seen:
            continue
        seen.add(key)
        ws = _words(d)
        nw = len(ws)
        if nw < 20:
            rest.append(i)
            continue
        # junk checks: very repetitive text, or text that is mostly not words
        if nw >= 100 and len(set(ws)) / nw < 0.12:
            rest.append(i)
            continue
        head_len = min(len(d), SCORE_CHARS)
        if sum(len(w) for w in ws) / head_len < 0.5:
            rest.append(i)
            continue
        s = 0.0
        for w in ws:
            s += wget(w, unk_w)
        scored.append((s / nw, i))

    scored.sort(key=lambda x: -x[0])

    # 4. Greedy fill with real token counts, best first, stop at the doc crossing T
    budget = float(T)
    chosen = []      # indices in score-descending order
    counts = []      # token count of each chosen doc
    used = 0.0

    def count_chunk(idxs):
        if enc is not None and (time.time() - t_start) < TIME_GUARD:
            try:
                toks = enc.encode_ordinary_batch(
                    [docs[i] for i in idxs], num_threads=TOK_THREADS
                )
                return [len(x) for x in toks]
            except Exception:
                pass
        return [len(docs[i]) * tpc for i in idxs]

    def fill(order):
        nonlocal used
        for start in range(0, len(order), TOK_CHUNK):
            if used >= budget:
                return
            chunk = order[start:start + TOK_CHUNK]
            cs = count_chunk(chunk)
            for i, c in zip(chunk, cs):
                c = max(1.0, float(c))
                chosen.append(i)
                counts.append(c)
                used += c
                if used >= budget:
                    return

    fill([i for _, i in scored])

    # Sanity fallback: make sure we have a plausible number of documents
    min_docs = int(0.25 * T / avg_tokens)
    if len(chosen) < min_docs and used < budget:
        fill(rest)

    if not chosen:
        return list(range(n))

    # 5. Three-block curriculum by score rank, split by cumulative tokens
    total = used
    hi_cut = HIGH_FRAC * total
    mid_cut = (1.0 - LOW_FRAC) * total
    high, mid, low = [], [], []
    cum = 0.0
    for i, c in zip(chosen, counts):
        if cum < hi_cut:
            high.append(i)
        elif cum < mid_cut:
            mid.append(i)
        else:
            low.append(i)
        cum += c

    rng.shuffle(low)
    rng.shuffle(mid)
    rng.shuffle(high)
    return low + mid + high
