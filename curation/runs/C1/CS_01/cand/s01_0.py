"""Curator: rank pool documents by vocabulary overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Score each document by word overlap with the target sample, using its
     first N_WORDS words: words absent from the target vocabulary are penalised,
     shared words are weighted by how much more frequent they are in the target
     sample than in the pool.
  3. Walk the documents from highest score down, counting tokens with the
     tiktoken "gpt2" encoder, until the budget of T tokens is filled.
  4. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
     mixed stream; the unchosen documents follow in score order as a backstop.
"""
import os
import re
import math
import random
import hashlib
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

N_WORDS = 400          # words per document used for scoring
CHAR_CAP = 6000        # characters per document scanned for those words
MISS_PENALTY = -1.0    # weight of a word that never occurs in the target sample
LLR_LO, LLR_HI = -2.0, 4.0
BUDGET_MARGIN = 1.03   # select slightly more than T to absorb count mismatches
CHUNK = 2000           # documents per tokeniser batch

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _words(text):
    return _WORD_RE.findall(text[:CHAR_CAP].lower())[:N_WORDS]


def _make_counter():
    """Return a function mapping a list of strings to a list of token counts."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            try:
                return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
            except Exception:
                out = []
                for t in texts:
                    try:
                        out.append(len(enc.encode_ordinary(t)))
                    except Exception:
                        out.append(max(1, len(t) // 4))
                return out

        return count
    except Exception:
        return lambda texts: [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- target statistics ----
    t_counts = Counter()
    t_lens = []
    for s in target_sample:
        w = _words(s) if isinstance(s, str) else []
        t_counts.update(w)
        t_lens.append(len(w))
    t_total = sum(t_counts.values())

    if t_total == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    t_lens.sort()
    # minimum words a document needs, adapted to how short target docs are
    min_words = max(3, min(20, (t_lens[len(t_lens) // 10] if t_lens else 20) // 2))

    # ---- pool pass: dedupe, tokenise into words, pool word counts ----
    seen = set()
    cand = []          # indices of candidate documents
    cand_words = []    # their word lists
    p_counts = Counter()
    for i, d in enumerate(docs):
        if not isinstance(d, str) or not d:
            continue
        key = hashlib.md5(
            _WS_RE.sub(" ", d).strip().lower().encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        w = _words(d)
        if len(w) < min_words:
            continue
        # highly repetitive documents (boilerplate / junk)
        if len(w) >= 50 and len(set(w)) / len(w) < 0.2:
            continue
        cand.append(i)
        cand_words.append(w)
        p_counts.update(w)
    p_total = max(1, sum(p_counts.values()))

    # ---- per-word weights ----
    weight = {}
    for w, pc in p_counts.items():
        tc = t_counts.get(w, 0)
        if tc == 0:
            weight[w] = MISS_PENALTY
        else:
            llr = math.log((tc / t_total) / (pc / p_total))
            weight[w] = min(LLR_HI, max(LLR_LO, llr))

    # ---- score documents ----
    scored = []
    for i, w in zip(cand, cand_words):
        s = sum(weight[x] for x in w) / len(w)
        scored.append((s, i))
    del cand_words
    # deterministic ordering: score descending, index as tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

    # ---- fill the token budget from the top of the ranking ----
    count = _make_counter()
    budget = int(T * BUDGET_MARGIN)
    chosen = []
    total = 0
    pos = 0
    while pos < len(ranked) and total < budget:
        batch = ranked[pos:pos + CHUNK]
        lens = count([docs[i] for i in batch])
        for i, L in zip(batch, lens):
            if total >= budget:
                break
            chosen.append(i)
            total += L
            pos += 1

    if not chosen:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    rest = ranked[len(chosen):]
    rng.shuffle(chosen)
    # If the best documents fall short of T, the remaining ones (still in score
    # order) top the stream up before any cycling happens.
    return chosen + rest
