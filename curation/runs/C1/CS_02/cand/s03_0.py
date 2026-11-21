"""Curator: rank pool documents by lexical similarity to the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by lexical overlap with the target sample:
   the Jaccard similarity (intersection over union) between its set of word
   bigrams and the target sample's set of word bigrams, plus a smaller
   contribution from the fraction of its words in the target vocabulary.
   The two components live on very different scales (Jaccard against a large
   target set is numerically tiny), so each is standardised across the pool
   before the weighted sum; the bigram Jaccard term carries the larger weight.
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
_W_BIGRAM = 1.0        # weight of the (standardised) bigram Jaccard similarity
_W_UNIGRAM = 0.25      # weight of the (standardised) unigram coverage


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


def _standardise(values):
    """Z-score a list of floats (returns zeros if there is no variance)."""
    m = len(values)
    if m == 0:
        return []
    mean = sum(values) / m
    var = sum((v - mean) ** 2 for v in values) / m
    std = var ** 0.5
    if std <= 1e-12:
        return [0.0] * m
    return [(v - mean) / std for v in values]


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
    n_tgt_bigrams = len(tgt_bigrams)

    # ---- 3. score pool documents ----
    seen = set()
    cand_idx = []     # pool index of each scorable doc
    cand_bi = []      # bigram Jaccard similarity with the target sample
    cand_uni = []     # fraction of words in the target vocabulary
    cand_junk = []    # True if the tokens-per-char ratio looks like junk
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

        # Jaccard similarity on bigram *sets*: |D ∩ T| / |D ∪ T|.
        doc_bigrams = set(zip(w, w[1:]))
        inter = sum(1 for b in doc_bigrams if b in tgt_bigrams)
        union = len(doc_bigrams) + n_tgt_bigrams - inter
        bi = inter / union if union > 0 else 0.0

        ratio = L / len(d)
        cand_idx.append(i)
        cand_bi.append(bi)
        cand_uni.append(uni)
        cand_junk.append(ratio > ratio_hi or ratio < ratio_lo)

    # Put both components on a common scale so the weights are meaningful,
    # then weight the bigram Jaccard term more heavily than the unigram term.
    z_bi = _standardise(cand_bi)
    z_uni = _standardise(cand_uni)

    scored = []      # (score, tie_break, index) for normal docs
    fallback = []    # docs that look like junk; only used if needed
    for k, i in enumerate(cand_idx):
        score = _W_BIGRAM * z_bi[k] + _W_UNIGRAM * z_uni[k]
        item = (score, rng.random(), i)
        if cand_junk[k]:
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
