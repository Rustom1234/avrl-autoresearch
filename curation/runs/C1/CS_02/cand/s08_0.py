"""Curator: density-filtered, diversity-aware top-K selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document gets a *document density*: the number of dictionary-like
   words per GPT-2 token. Clean prose has a high density (most tokens are
   whole words or parts of words); noisy text (symbol runs, encoding damage,
   base64, markup debris, tables of numbers) spends many tokens on non-word
   characters and has a low density. Thresholds are calibrated on the target
   sample:
     - documents below a hard floor are removed from the pool (kept only as
       a fallback if nothing else is available);
     - documents between the hard floor and a soft floor have their
       similarity score multiplied by a penalty that falls smoothly to 0.
4. Each pool document is scored by lexical overlap with the target sample:
   the fraction of its word bigrams that occur in the target sample, plus a
   small contribution from the fraction of its words in the target vocabulary,
   times the density penalty. Documents whose tokens-per-character ratio is far
   outside the target range are also only used as a fallback.
5. The top 15% of the pool (by document count) under that score forms the
   candidate set.
6. Within the candidate set, documents are picked greedily (lazy greedy with a
   heap) to maximise the number of *new* unique target bigrams covered, with
   the gain normalised by sqrt(token length). When no candidate adds new
   coverage, the covered set is reset and another round starts. Selection
   stops at T tokens or when the candidates are exhausted; a list shorter than
   T tokens is cycled by the trainer.
7. The selected set is shuffled (seeded by CURATE_SEED) so that training does
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

# Density calibration (relative to the target sample's density distribution).
_SOFT_PCTL = 0.10      # target percentile below which the penalty starts
_HARD_FACTOR = 0.65    # hard floor = _HARD_FACTOR * soft floor
_PENALTY_POWER = 2.0   # sharpness of the penalty between hard and soft floor


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


def _density(n_words, n_tokens, n_chars):
    """Words per token.

    Words are counted on the first _SCORE_CHARS characters only, so the token
    count is scaled to the same prefix.
    """
    if n_tokens <= 0 or n_chars <= 0:
        return 0.0
    frac = min(1.0, _SCORE_CHARS / float(n_chars))
    tok = max(1.0, n_tokens * frac)
    return n_words / tok


def _density_thresholds(target_sample, tgt_len):
    """(hard_floor, soft_floor) for document density, calibrated on the target."""
    dens = []
    for t, tl in zip(target_sample, tgt_len):
        if not t or tl <= 0:
            continue
        w = _words(t)
        if len(w) < _MIN_WORDS:
            continue
        dens.append(_density(len(w), tl, len(t)))
    if not dens:
        return 0.0, 0.0
    dens.sort()
    soft = dens[min(len(dens) - 1, int(_SOFT_PCTL * len(dens)))]
    # Never let the hard floor exceed the lowest density seen in the target.
    hard = min(_HARD_FACTOR * soft, dens[0])
    return hard, soft


def _density_penalty(density, hard, soft):
    """1.0 at or above the soft floor, falling smoothly to 0.0 at the hard floor."""
    if soft <= 0.0 or density >= soft:
        return 1.0
    if density <= hard or soft <= hard:
        return 0.0
    x = (density - hard) / (soft - hard)
    return x ** _PENALTY_POWER


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

    # Document-density (words per token) thresholds from the target.
    dens_hard, dens_soft = _density_thresholds(target_sample, tgt_len)

    # ---- 2. target lexical profile ----
    tgt_vocab = set()
    tgt_bigrams = {}   # bigram -> integer id
    for t in target_sample:
        w = _words(t)
        tgt_vocab.update(w)
        for b in zip(w, w[1:]):
            if b not in tgt_bigrams:
                tgt_bigrams[b] = len(tgt_bigrams)

    # ---- 3. score pool documents (with density filter / penalty) ----
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
        uni = sum(1 for x in w if x in tgt_vocab) / nw
        ids = []
        for b in zip(w, w[1:]):
            j = get(b)
            if j is not None:
                ids.append(j)
        bi = len(ids) / (nw - 1)
        base = bi + 0.25 * uni

        density = _density(nw, L, len(d))
        penalty = _density_penalty(density, dens_hard, dens_soft)
        score = base * penalty

        ratio = L / len(d)
        low_density = dens_soft > 0.0 and density <= dens_hard
        if low_density or ratio > ratio_hi or ratio < ratio_lo:
            # Rank fallback docs by density first, so the least noisy come first.
            fallback.append((base * min(1.0, density / dens_soft) if dens_soft > 0 else base,
                             rng.random(), i))
        else:
            scored.append((score, rng.random(), i))
        cover[i] = frozenset(ids)

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
    selected = _greedy_coverage(cands, cover, doc_len, T)

    if not selected:
        selected = [i for _, _, i in cands]

    # ---- 6. shuffle so training does not see a coverage-sorted curriculum ----
    rng.shuffle(selected)
    return selected
