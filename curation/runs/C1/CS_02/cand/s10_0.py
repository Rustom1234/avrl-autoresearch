"""Curator: diversity-aware top-K selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by
       score = similarity * density
   where
       similarity = shared / (shared + 0.1 * target_sample_bigrams)
           shared                = number of distinct target-sample bigrams
                                   that occur in the document,
           target_sample_bigrams = number of distinct word bigrams in the
                                   target sample,
       density    = unique bigrams in the document / total bigrams in the document
   ("information density": documents that keep repeating the same patterns
   get a low value, documents packed with distinct content a high one).
   Documents whose tokens-per-character ratio is far outside the target range
   (typical of damaged / junk text) are only used if nothing else is available.
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

import tiktoken

from common import T

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WS_RE = re.compile(r"\s+")
_SCORE_CHARS = 20000   # only the first chars of each doc are used for scoring (speed)
_MIN_WORDS = 8         # docs with fewer words cannot be scored reliably
_CHUNK = 2000          # docs per tiktoken batch (keeps memory bounded)
_TOP_FRACTION = 0.15   # fraction of the pool kept as candidates
_SMOOTH = 0.1          # weight of the target-sample bigram count in the similarity


def _token_lengths(enc, texts):
    """Token count of each text, computed in batches so token lists are not retained."""
    lengths = []
    for start in range(0, len(texts), _CHUNK):
        chunk = texts[start:start + _CHUNK]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=8)
            lengths.extend(len(e) for e in encoded)
        except Exception:
            for t in chunk:
                try:
                    lengths.append(len(enc.encode_ordinary(t)))
                except Exception:
                    lengths.append(max(1, len(t) // 4))
    return lengths


def _words(text):
    return _WORD_RE.findall(text[:_SCORE_CHARS].lower())


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

    # ---- 1. token lengths (pool and target sample) ----
    enc = tiktoken.get_encoding("gpt2")
    doc_len = _token_lengths(enc, docs)
    tgt_len = _token_lengths(enc, list(target_sample))

    # Tokens-per-character statistics of the target, used to flag junk text.
    ratios = sorted(
        tl / len(t) for tl, t in zip(tgt_len, target_sample) if len(t) > 0 and tl > 0
    )
    if ratios:
        ratio_hi = ratios[-1] * 1.5
        ratio_lo = ratios[0] / 2.0
    else:
        ratio_hi, ratio_lo = float("inf"), 0.0

    # ---- 2. target lexical profile ----
    tgt_bigrams = {}   # bigram -> integer id
    for t in target_sample:
        w = _words(t)
        for b in zip(w, w[1:]):
            if b not in tgt_bigrams:
                tgt_bigrams[b] = len(tgt_bigrams)
    smooth = _SMOOTH * max(1, len(tgt_bigrams))

    # ---- 3. score pool documents ----
    seen = set()
    scored = []      # (score, tie_break, index) for normal docs
    fallback = []    # docs that look like junk; only used if needed
    cover = {}       # index -> frozenset of target-bigram ids in the doc
    get = tgt_bigrams.get
    for i, d in enumerate(docs):
        L = doc_len[i]
        if L <= 0 or not d:
            continue
        key = hash(_WS_RE.sub(" ", d).strip().lower())
        if key in seen:
            continue
        seen.add(key)

        w = _words(d)
        nw = len(w)
        if nw < _MIN_WORDS:
            continue

        total_bigrams = nw - 1
        uniq = set(zip(w, w[1:]))
        ids = set()
        for b in uniq:
            j = get(b)
            if j is not None:
                ids.add(j)

        shared = len(ids)
        similarity = shared / (shared + smooth)
        density = len(uniq) / total_bigrams
        score = similarity * density

        ratio = L / len(d)
        item = (score, rng.random(), i)
        if ratio > ratio_hi or ratio < ratio_lo:
            fallback.append(item)
        else:
            scored.append(item)
        cover[i] = frozenset(ids)

    scored.sort(reverse=True)
    fallback.sort(reverse=True)

    # ---- 4. candidate set: top 15% of the pool by score ----
    k = max(1, int(math.ceil(_TOP_FRACTION * n)))
    cands = scored[:k]
    if len(cands) < k:
        cands = cands + fallback[:k - len(cands)]

    if not cands:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- 5. greedy coverage of unique target bigrams within the candidates ----
    selected = _greedy_coverage(cands, cover, doc_len, T)

    if not selected:
        selected = [i for _, _, i in cands]

    # ---- 6. shuffle so training does not see a coverage-sorted curriculum ----
    rng.shuffle(selected)
    return selected
