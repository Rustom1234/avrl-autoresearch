"""Curator: rank pool documents by lexical similarity to the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by lexical overlap with the target sample:
   the fraction of its word bigrams that occur in the target sample, plus a
   small contribution from the fraction of its words in the target vocabulary.
   Documents whose tokens-per-character ratio is far outside the target range
   (typical of damaged / junk text) are pushed to the back.
4. The most similar documents are selected first until the budget of T tokens
   is filled; the selected set is then shuffled (seeded by CURATE_SEED) so that
   training does not see a similarity-sorted curriculum.
"""
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
    tgt_vocab = set()
    tgt_bigrams = set()
    for t in target_sample:
        w = _words(t)
        tgt_vocab.update(w)
        tgt_bigrams.update(zip(w, w[1:]))

    # ---- 3. score pool documents ----
    seen = set()
    scored = []      # (score, tie_break, index) for normal docs
    fallback = []    # docs that look like junk; only used if needed
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
        bi = sum(1 for b in zip(w, w[1:]) if b in tgt_bigrams) / (nw - 1)
        score = bi + 0.25 * uni

        ratio = L / len(d)
        item = (score, rng.random(), i)
        if ratio > ratio_hi or ratio < ratio_lo:
            fallback.append(item)
        else:
            scored.append(item)

    scored.sort(reverse=True)
    fallback.sort(reverse=True)

    # ---- 4. take the most similar documents first, up to T tokens ----
    selected = []
    total = 0
    for group in (scored, fallback):
        for _, _, i in group:
            if total >= T:
                break
            selected.append(i)
            total += doc_len[i]
        if total >= T:
            break

    if not selected:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    rng.shuffle(selected)
    return selected
