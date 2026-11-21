"""Curator: diversity-aware top-K selection on GPT-2 subword bigrams.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Every pool and target document is tokenised with tiktoken ("gpt2"), the
   tokenizer of the trained model. Token lengths come from the same pass.
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by subword overlap with the target sample:
   the fraction of its token bigrams that occur in the target sample, plus a
   small contribution from the fraction of its tokens in the target token
   vocabulary. Documents whose tokens-per-character ratio is far outside the
   target range (typical of damaged / junk text) are only used if nothing
   else is available.
4. The top 15% of the pool (by document count) under that score forms the
   candidate set.
5. Within the candidate set, documents are picked greedily (lazy greedy with a
   heap) to maximise the number of *new* unique target token bigrams covered,
   with the gain normalised by sqrt(token length). When no candidate adds new
   coverage, the covered set is reset and another round starts, so later
   rounds again favour broad coverage. Selection stops at T tokens or when
   the candidates are exhausted; a list shorter than T tokens is cycled by
   the trainer.
6. The selected set is shuffled (seeded by CURATE_SEED) so that training does
   not see a coverage-sorted curriculum.
"""
import heapq
import math
import os
import random
import re

import numpy as np
import tiktoken

from common import T

_WS_RE = re.compile(r"\s+")
_SCORE_TOKENS = 5000   # only the first tokens of each doc are used for scoring (speed)
_MIN_TOKENS = 16       # docs with fewer tokens cannot be scored reliably
_CHUNK = 2000          # docs per tiktoken batch (keeps memory bounded)
_TOP_FRACTION = 0.15   # fraction of the pool kept as candidates


def _encode_chunk(enc, chunk):
    """Token lists for a chunk of texts; None for a text that cannot be encoded."""
    try:
        return enc.encode_ordinary_batch(chunk, num_threads=8)
    except Exception:
        out = []
        for t in chunk:
            try:
                out.append(enc.encode_ordinary(t))
            except Exception:
                out.append(None)
        return out


def _bigram_keys(arr, vocab_size):
    """Pack consecutive token pairs into single int64 keys."""
    return arr[:-1] * vocab_size + arr[1:]


def _greedy_coverage(cands, cover, doc_len, budget, n_bigrams):
    """Order candidates by greedy marginal coverage of target token bigrams.

    cands: list of (score, tie, index), sorted by score descending.
    cover: dict index -> numpy array of unique target-bigram ids in the doc.
    Returns the list of picked indices (total tokens just reaching `budget`,
    or all candidates if they hold fewer tokens).
    """
    selected = []
    total = 0
    remaining = [i for _, _, i in cands]            # in score order
    rank = {i: r for r, i in enumerate(remaining)}  # tie-break: higher score first
    norm = {i: math.sqrt(max(1, doc_len[i])) for i in remaining}

    while remaining and total < budget:
        # Start a new coverage round over the documents not yet picked.
        covered = np.zeros(max(1, n_bigrams), dtype=bool)
        heap = []
        for i in remaining:
            g = cover[i].size
            if g > 0:
                heap.append((-g / norm[i], rank[i], i))
        if not heap:
            # Nothing left covers any target bigram: fall back to score order.
            for i in remaining:
                if total >= budget:
                    break
                selected.append(i)
                total += doc_len[i]
            return selected
        heapq.heapify(heap)

        picked = set()
        while heap and total < budget:
            _, r, i = heapq.heappop(heap)
            s = cover[i]
            g = s.size - int(np.count_nonzero(covered[s]))
            if g <= 0:
                continue  # no new coverage this round; stays for the next round
            prio = -g / norm[i]
            if heap and prio > heap[0][0]:
                # Stale estimate: another doc may now be better.
                heapq.heappush(heap, (prio, r, i))
                continue
            selected.append(i)
            picked.add(i)
            total += doc_len[i]
            covered[s] = True

        if not picked:
            break
        remaining = [i for i in remaining if i not in picked]

    return selected


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = int(enc.n_vocab)

    # ---- 1. target subword profile (token vocabulary and token bigrams) ----
    targets = list(target_sample)
    tgt_len = []
    vocab_mask = np.zeros(vocab_size, dtype=bool)
    key_parts = []
    for start in range(0, len(targets), _CHUNK):
        chunk = targets[start:start + _CHUNK]
        for t, toks in zip(chunk, _encode_chunk(enc, chunk)):
            if toks is None:
                tgt_len.append(0)
                continue
            tgt_len.append(len(toks))
            if not toks:
                continue
            arr = np.asarray(toks[:_SCORE_TOKENS], dtype=np.int64)
            vocab_mask[arr] = True
            if arr.size > 1:
                key_parts.append(_bigram_keys(arr, vocab_size))
    if key_parts:
        tgt_keys = np.unique(np.concatenate(key_parts))
    else:
        tgt_keys = np.zeros(0, dtype=np.int64)
    n_bigrams = int(tgt_keys.size)
    del key_parts

    # Tokens-per-character statistics of the target, used to flag junk text.
    ratios = sorted(
        tl / len(t) for tl, t in zip(tgt_len, targets) if len(t) > 0 and tl > 0
    )
    if ratios:
        ratio_hi = ratios[-1] * 1.5
        ratio_lo = ratios[0] / 2.0
    else:
        ratio_hi, ratio_lo = float("inf"), 0.0

    # ---- 2. tokenise and score pool documents (chunked; tokens not retained) ----
    doc_len = [0] * n
    seen = set()
    scored = []      # (score, tie_break, index) for normal docs
    fallback = []    # docs that look like junk; only used if needed
    cover = {}       # index -> unique target-bigram ids in the doc (int32 array)
    empty_ids = np.zeros(0, dtype=np.int32)

    for start in range(0, n, _CHUNK):
        chunk = docs[start:start + _CHUNK]
        encoded = _encode_chunk(enc, chunk)
        for off, (d, toks) in enumerate(zip(chunk, encoded)):
            i = start + off
            if toks is None:
                doc_len[i] = max(1, len(d) // 4)
                continue
            L = len(toks)
            doc_len[i] = L
            if L <= 0 or not d:
                continue
            key = hash(_WS_RE.sub(" ", d).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            if L < _MIN_TOKENS:
                continue

            arr = np.asarray(toks[:_SCORE_TOKENS], dtype=np.int64)
            nt = arr.size
            uni = float(np.count_nonzero(vocab_mask[arr])) / nt
            if n_bigrams > 0:
                keys = _bigram_keys(arr, vocab_size)
                pos = np.searchsorted(tgt_keys, keys)
                np.minimum(pos, n_bigrams - 1, out=pos)
                hit = pos[tgt_keys[pos] == keys]
                bi = hit.size / (nt - 1)
                ids = np.unique(hit).astype(np.int32)
            else:
                bi = 0.0
                ids = empty_ids
            score = bi + 0.25 * uni

            ratio = L / len(d)
            item = (score, rng.random(), i)
            if ratio > ratio_hi or ratio < ratio_lo:
                fallback.append(item)
            else:
                scored.append(item)
            cover[i] = ids
        del encoded

    scored.sort(reverse=True)
    fallback.sort(reverse=True)

    # ---- 3. candidate set: top 15% of the pool by similarity ----
    k = max(1, int(math.ceil(_TOP_FRACTION * n)))
    cands = scored[:k]
    if len(cands) < k:
        cands = cands + fallback[:k - len(cands)]

    if not cands:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- 4. greedy coverage of unique target token bigrams within the candidates ----
    cand_cover = {i: cover[i] for _, _, i in cands}
    del cover
    selected = _greedy_coverage(cands, cand_cover, doc_len, T, n_bigrams)

    if not selected:
        selected = [i for _, _, i in cands]

    # ---- 5. shuffle so training does not see a coverage-sorted curriculum ----
    rng.shuffle(selected)
    return selected
