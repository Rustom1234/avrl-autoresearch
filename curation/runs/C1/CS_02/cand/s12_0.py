"""Curator: diversity-aware top-K selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Every pool and target document is tokenised with tiktoken ("gpt2"); the
   token ids give both the token length and the similarity features.
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by subword overlap with the target sample:
   the fraction of its token-id bigrams that occur in the target sample, plus a
   small contribution from the fraction of its tokens in the target token
   vocabulary. Documents whose tokens-per-character ratio is far outside the
   target range (typical of damaged / junk text) are only used if nothing else
   is available.
4. The top 15% of the pool (by document count) under that score forms the
   candidate set.
5. Within the candidate set, documents are picked greedily (lazy greedy with a
   heap) to maximise the number of *new* unique target bigrams covered, with
   the gain normalised by sqrt(token length). When no candidate adds new
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
_MIN_TOKENS = 8        # docs with fewer tokens cannot be scored reliably
_CHUNK = 2000          # docs per tiktoken batch (keeps memory bounded)
_TOP_FRACTION = 0.15   # fraction of the pool kept as candidates


def _encode_chunk(enc, chunk):
    """Token-id lists for a chunk of texts.

    Returns a list whose entries are lists of ids, or None when a text could
    not be encoded at all.
    """
    try:
        return enc.encode_batch(chunk, num_threads=8, disallowed_special=())
    except Exception:
        out = []
        for t in chunk:
            try:
                out.append(enc.encode(t, disallowed_special=()))
            except Exception:
                out.append(None)
        return out


def _bigram_keys(arr, vocab_size):
    """Pack consecutive token-id pairs into single int64 keys."""
    return arr[:-1] * np.int64(vocab_size) + arr[1:]


def _greedy_coverage(cands, cover, doc_len, budget):
    """Order candidates by greedy marginal coverage of target bigrams.

    cands: list of (score, tie, index), sorted by score descending.
    cover: dict index -> frozenset of target-bigram ids contained in the doc.
    Returns the list of picked indices (total tokens just reaching `budget`,
    or all candidates if they hold fewer tokens).
    """
    selected = []
    total = 0
    remaining = [i for _, _, i in cands]          # in score order
    rank = {i: r for r, i in enumerate(remaining)}  # tie-break: higher score first
    norm = {i: math.sqrt(max(1, doc_len[i])) for i in remaining}

    while remaining and total < budget:
        # Start a new coverage round over the documents not yet picked.
        covered = set()
        heap = []
        for i in remaining:
            g = len(cover[i])
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
            g = sum(1 for b in s if b not in covered)
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
            covered.update(s)

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
    vocab_size = max(int(getattr(enc, "n_vocab", 50257)), 50257)

    # ---- 1. target profile: token lengths, token vocabulary, token bigrams ----
    target_sample = list(target_sample)
    tgt_len = []
    vocab_mask = np.zeros(vocab_size, dtype=bool)
    key_parts = []
    for start in range(0, len(target_sample), _CHUNK):
        chunk = target_sample[start:start + _CHUNK]
        for t, toks in zip(chunk, _encode_chunk(enc, chunk)):
            if toks is None:
                tgt_len.append(max(1, len(t) // 4) if t else 0)
                continue
            tgt_len.append(len(toks))
            arr = np.asarray(toks[:_SCORE_TOKENS], dtype=np.int64)
            if arr.size:
                vocab_mask[arr] = True
            if arr.size >= 2:
                key_parts.append(_bigram_keys(arr, vocab_size))
    if key_parts:
        tgt_keys = np.unique(np.concatenate(key_parts))   # sorted; position = bigram id
    else:
        tgt_keys = np.zeros(0, dtype=np.int64)
    n_tgt = int(tgt_keys.size)

    # Tokens-per-character statistics of the target, used to flag junk text.
    ratios = sorted(
        tl / len(t) for tl, t in zip(tgt_len, target_sample) if len(t) > 0 and tl > 0
    )
    if ratios:
        ratio_hi = ratios[-1] * 1.5
        ratio_lo = ratios[0] / 2.0
    else:
        ratio_hi, ratio_lo = float("inf"), 0.0

    # ---- 2./3. tokenise the pool chunk by chunk, dedup and score ----
    doc_len = [0] * n
    seen = set()
    scored = []      # (score, tie_break, index) for normal docs
    fallback = []    # docs that look like junk; only used if needed
    cover_arr = {}   # index -> sorted unique array of target-bigram ids in the doc
    empty_ids = np.zeros(0, dtype=np.int32)

    for start in range(0, n, _CHUNK):
        chunk = docs[start:start + _CHUNK]
        encoded = _encode_chunk(enc, chunk)
        for off, (d, toks) in enumerate(zip(chunk, encoded)):
            i = start + off
            if not d:
                continue
            if toks is None:
                doc_len[i] = max(1, len(d) // 4)
                continue  # cannot be scored
            L = len(toks)
            doc_len[i] = L
            if L <= 0:
                continue
            key = hash(_WS_RE.sub(" ", d).strip().lower())
            if key in seen:
                continue
            seen.add(key)

            if L < _MIN_TOKENS:
                continue
            arr = np.asarray(toks[:_SCORE_TOKENS], dtype=np.int64)
            uni = float(vocab_mask[arr].mean())
            if n_tgt:
                keys = _bigram_keys(arr, vocab_size)
                pos = np.searchsorted(tgt_keys, keys)
                np.minimum(pos, n_tgt - 1, out=pos)
                match = tgt_keys[pos] == keys
                bi = float(match.mean())
                ids = np.unique(pos[match]).astype(np.int32)
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
            cover_arr[i] = ids
        del encoded

    scored.sort(reverse=True)
    fallback.sort(reverse=True)

    # ---- 4. candidate set: top 15% of the pool by similarity ----
    k = max(1, int(math.ceil(_TOP_FRACTION * n)))
    cands = scored[:k]
    if len(cands) < k:
        cands = cands + fallback[:k - len(cands)]

    if not cands:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- 5. greedy coverage of unique target bigrams within the candidates ----
    cover = {i: frozenset(cover_arr[i].tolist()) for _, _, i in cands}
    del cover_arr
    selected = _greedy_coverage(cands, cover, doc_len, T)

    if not selected:
        selected = [i for _, _, i in cands]

    # ---- 6. shuffle so training does not see a coverage-sorted curriculum ----
    rng.shuffle(selected)
    return selected
