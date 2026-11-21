"""Curator: rank pool documents by lexical similarity to the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by lexical overlap with the target sample:
     score = 1.0  * fraction of its word bigrams that occur in the target sample
           + 0.1  * fraction of its words that are in the target vocabulary
           + 0.25 * rare-word score
           + 0.15 * punctuation score
   - rare-word score: IDF-weighted share of the document's distinct words that
     are "domain words" (words seen in at least two target documents). IDF is
     computed over the pool, so words that are rare in the pool but recurrent
     in the target count the most.
   - punctuation score: how well the document reproduces the punctuation marks
     that are over-represented in the target relative to the pool as a whole.
   Documents whose tokens-per-character ratio is far outside the target range
   (typical of damaged / junk text) are pushed to the back.
4. The most similar documents are selected first until the budget of T tokens
   is filled; the selected set is then shuffled (seeded by CURATE_SEED) so that
   training does not see a similarity-sorted curriculum.
"""
import math
import os
import random
import re
from collections import Counter

import numpy as np
import tiktoken

from common import T

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WS_RE = re.compile(r"\s+")
_SCORE_CHARS = 20000   # only the first chars of each doc are used for scoring (speed)
_MIN_WORDS = 8         # docs with fewer words cannot be scored reliably
_CHUNK = 2000          # docs per tiktoken batch (keeps memory bounded)

_W_BIGRAM = 1.0
_W_UNIGRAM = 0.1
_W_RARE = 0.25
_W_PUNCT = 0.15

_MAX_PUNCT = 24        # number of distinct target punctuation marks tracked
_MIN_PUNCT_COUNT = 5   # a mark must occur this often in the target sample
_RARE_MIN_TGT_DF = 2   # a "domain word" must occur in this many target docs


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
    tgt_df = Counter()          # number of target docs containing each word
    punct_counter = Counter()   # punctuation / symbol characters in the target
    tgt_chars = 0
    for t in target_sample:
        head = t[:_SCORE_CHARS]
        w = _WORD_RE.findall(head.lower())
        tgt_vocab.update(w)
        tgt_bigrams.update(zip(w, w[1:]))
        tgt_df.update(set(w))
        tgt_chars += len(head)
        for ch in head:
            if not ch.isalnum() and not ch.isspace():
                punct_counter[ch] += 1

    # Domain words: recurrent in the target (not one-off names / typos).
    rare_id = {}
    for word in sorted(tgt_df):
        if tgt_df[word] >= _RARE_MIN_TGT_DF:
            rare_id[word] = len(rare_id)
    n_rare = len(rare_id)

    # Punctuation marks tracked (most frequent in the target).
    pchars = [
        c for c, k in sorted(punct_counter.items(), key=lambda x: (-x[1], x[0]))
        if k >= _MIN_PUNCT_COUNT
    ][:_MAX_PUNCT]
    n_punct = len(pchars)

    # ---- 3. first pass: per-document raw statistics ----
    seen = set()
    idx_list = []       # pool index of each scored doc
    base_list = []      # bigram + unigram part of the score
    junk_list = []      # True if the tokens-per-char ratio looks wrong
    tie_list = []
    distinct_list = []  # number of distinct words in the doc
    rare_ids = []       # per doc: int32 array of domain-word ids present
    punct_rows = []     # per doc: counts of each tracked punctuation mark
    head_lens = []

    for i, d in enumerate(docs):
        L = doc_len[i]
        if L <= 0 or not d:
            continue
        key = hash(_WS_RE.sub(" ", d).strip().lower())
        if key in seen:
            continue
        seen.add(key)

        head = d[:_SCORE_CHARS]
        w = _WORD_RE.findall(head.lower())
        nw = len(w)
        if nw < _MIN_WORDS:
            continue
        uni = sum(1 for x in w if x in tgt_vocab) / nw
        bi = sum(1 for b in zip(w, w[1:]) if b in tgt_bigrams) / (nw - 1)

        distinct = set(w)
        ids = [rare_id[x] for x in distinct if x in rare_id]

        ratio = L / len(d)
        idx_list.append(i)
        base_list.append(_W_BIGRAM * bi + _W_UNIGRAM * uni)
        junk_list.append(ratio > ratio_hi or ratio < ratio_lo)
        tie_list.append(rng.random())
        distinct_list.append(len(distinct))
        rare_ids.append(np.array(ids, dtype=np.int32))
        punct_rows.append([head.count(c) for c in pchars])
        head_lens.append(len(head))

    m = len(idx_list)
    if m == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    score = np.array(base_list, dtype=np.float64)

    # ---- 3b. rare-word score (IDF over the pool, restricted to domain words) ----
    if n_rare > 0:
        counts = np.array([len(a) for a in rare_ids], dtype=np.int64)
        if counts.sum() > 0:
            all_ids = np.concatenate(rare_ids)
            df = np.bincount(all_ids, minlength=n_rare).astype(np.float64)
            idf = np.log((m + 1.0) / (df + 1.0)) / math.log(m + 1.0)
            owner = np.repeat(np.arange(m), counts)
            idf_sum = np.bincount(owner, weights=idf[all_ids], minlength=m)
            rare_score = idf_sum / np.maximum(np.array(distinct_list, dtype=np.float64), 1.0)
            score += _W_RARE * rare_score
    del rare_ids

    # ---- 3c. punctuation score (marks over-represented in the target) ----
    if n_punct > 0 and tgt_chars > 0:
        P = np.array(punct_rows, dtype=np.float64).reshape(m, n_punct)
        lens = np.maximum(np.array(head_lens, dtype=np.float64), 1.0)
        eps = 1e-6
        tgt_rate = np.array([punct_counter[c] for c in pchars], dtype=np.float64) / tgt_chars
        pool_rate = P.sum(axis=0) / lens.sum()
        weight = np.clip(np.log((tgt_rate + eps) / (pool_rate + eps)), 0.0, 3.0)
        wsum = weight.sum()
        if wsum > 0:
            doc_rate = P / lens[:, None]
            match = np.minimum(doc_rate / np.maximum(tgt_rate, eps), 1.0)
            punct_score = (match * weight).sum(axis=1) / wsum
            score += _W_PUNCT * punct_score
    del punct_rows

    # ---- 3d. split normal / junk-looking docs and sort by score ----
    scored = []
    fallback = []
    for k in range(m):
        item = (float(score[k]), tie_list[k], idx_list[k])
        if junk_list[k]:
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
