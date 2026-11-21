"""Curator: rank pool documents by vocabulary overlap and word-order agreement
with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Count the tokens of every pool document with tiktoken ("gpt2").
  2. Score every document with a target-aware TF-IDF overlap: sublinear term
     frequency times pool IDF, where
       - words of the target vocabulary count positively,
       - the top 50% most frequent target words (by the share of target
         documents that contain them) get an extra boost that grows with
         that share,
       - words that occur in many pool documents but in few (or no) target
         documents get a negative weight.
     The score is the weighted share of the document's word mass, floored
     at 0.
  3. Score every document by word order: the share of its adjacent
     whitespace-token bigrams that also occur in target_sample
     (bigram_hit = hits / (bigrams + 20)).  Word-shuffled articles keep their
     vocabulary but lose their bigrams, so they score low here.
  4. Calibrate a bigram_hit threshold without labels: score the real target
     documents and word-shuffled copies of them (leave-one-out bigram set) and
     take the midpoint between the 5th percentile of the real scores and the
     95th percentile of the shuffled scores.
  5. Mark as unusable: documents below the threshold, documents where more
     than 3% of adjacent tokens are punctuation pairs (', ,', '. ,', ...),
     exact duplicates and empty documents.
  6. Rank the usable documents by overlap score * bigram_hit, pick the best
     until the training budget of T tokens is filled, then shuffle the picked
     set with the seed from CURATE_SEED so training does not see a sorted
     stream.  If the usable documents hold fewer than T tokens, only they are
     returned and the trainer cycles the list.
"""
import os
import random
import re
from collections import Counter

import numpy as np

from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")

_BIGRAM_SHRINK = 20.0
_PUNCT_TOKENS = frozenset([",", ".", ";", ":", "!", "?"])
_PUNCT_PAIR_MAX = 0.03

# Target-aware TF-IDF parameters.
_TOP_TARGET_SHARE = 0.5     # share of target words (most frequent first) that get a boost
_BOOST = 1.0                # maximum extra weight of a boosted target word
_PENALTY = 1.0              # maximum negative weight (in idf units) of a pool-only word
_PENALTY_MIN_POOL_DF = 0.01  # "many pool documents": at least this share of the pool
_PENALTY_FULL_RATIO = 8.0   # pool share / target share at which the penalty is full


def _token_counts(docs):
    """Number of gpt2 tokens in each document."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        counts = np.empty(len(docs), dtype=np.int64)
        chunk = 2000
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            try:
                encoded = enc.encode_ordinary_batch(batch, num_threads=8)
                lens = [len(e) for e in encoded]
            except Exception:
                lens = []
                for d in batch:
                    try:
                        lens.append(len(enc.encode_ordinary(d)))
                    except Exception:
                        lens.append(max(1, len(d) // 4))
            counts[start:start + len(batch)] = lens
        return counts
    except Exception:
        return np.array([max(1, len(d) // 4) for d in docs], dtype=np.int64)


def _overlap_scores(docs, target_sample):
    """Target-aware TF-IDF overlap of each document with the target vocabulary.

    Per-word weight (applied to sublinear term frequency):
        idf_pool * ( in_target * (1 + boost) - penalty )
    boost   : only for the top 50% most frequent target words (ranked by the
              share of target documents containing them); proportional to
              that share relative to the most widespread target word.
    penalty : for words that are in many pool documents but in few target
              documents; grows with log(pool share / target share).
    The score is the weighted sum divided by the plain idf mass of the
    document (plus shrinkage), floored at 0.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    n = len(docs)
    vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN, dtype=np.float32)
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:
        return np.zeros(n, dtype=np.float64)

    n_feat = X.shape[1]
    vocab = vec.vocabulary_

    # Target statistics: document frequency and total count of every word.
    word_re = re.compile(_TOKEN_PATTERN)
    target_df = Counter()
    target_tf = Counter()
    n_target = 0
    for t in target_sample:
        words = word_re.findall(t.lower())
        if not words:
            continue
        n_target += 1
        target_tf.update(words)
        target_df.update(set(words))
    n_target = max(n_target, 1)

    # The top 50% most frequent target words (by document share, ties broken
    # by total count, then alphabetically for determinism).
    ranked = sorted(target_df, key=lambda w: (-target_df[w], -target_tf[w], w))
    n_top = int(np.ceil(len(ranked) * _TOP_TARGET_SHARE))
    top_words = set(ranked[:n_top])
    max_tdf = float(target_df[ranked[0]]) if ranked else 1.0

    mask = np.zeros(n_feat, dtype=np.float64)        # word is in the target vocabulary
    boost = np.zeros(n_feat, dtype=np.float64)       # extra weight for widespread target words
    t_share = np.zeros(n_feat, dtype=np.float64)     # share of target docs containing the word
    for w, c in target_df.items():
        j = vocab.get(w)
        if j is None:
            continue
        mask[j] = 1.0
        t_share[j] = c / n_target
        if w in top_words:
            boost[j] = _BOOST * (c / max_tdf)

    # Pool document frequency -> idf.
    df = np.bincount(X.indices, minlength=n_feat).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    p_share = df / max(n, 1)

    # Penalty: common in the pool, rare (or absent) in the target sample.
    eps = 0.5 / n_target
    log_ratio = np.log((p_share + eps) / (t_share + eps))
    penalty = np.clip(log_ratio / np.log(_PENALTY_FULL_RATIO), 0.0, 1.0)
    penalty[p_share < _PENALTY_MIN_POOL_DF] = 0.0
    penalty *= _PENALTY

    weight = idf * (mask * (1.0 + boost) - penalty)

    # Sublinear term frequency: repeated words do not dominate.
    X.data = np.log1p(X.data)

    num = np.asarray(X @ weight).ravel()
    den = np.asarray(X @ idf).ravel()
    # Small shrinkage so very short documents cannot get a perfect score.
    scores = num / (den + 30.0)
    return np.maximum(scores, 0.0)


def _ws_tokens(text):
    """Lowercased whitespace tokens; punctuation tokens (',', '.', '@-@') are kept."""
    return text.lower().split()


def _shuffled_tokens(text, rng):
    """Tokens of the document with everything after the title line shuffled."""
    stripped = text.lstrip()
    head, sep, body = stripped.partition("\n")
    if sep:
        head_toks = _ws_tokens(head)
        body_toks = _ws_tokens(body)
    else:
        head_toks = []
        body_toks = _ws_tokens(stripped)
    rng.shuffle(body_toks)
    return head_toks + body_toks


def _calibrate(target_sample, seed):
    """Return (target bigram set, bigram_hit threshold).

    Real and word-shuffled target documents are both scored against a
    leave-one-out bigram set (bigrams seen in at least one *other* target
    document).  The threshold is the midpoint between the 5th percentile of the
    real scores and the 95th percentile of the shuffled scores.
    """
    tok_lists = [_ws_tokens(t) for t in target_sample]
    doc_sets = [set(zip(toks, toks[1:])) for toks in tok_lists]

    doc_count = Counter()
    for s in doc_sets:
        doc_count.update(s)
    bigram_set = set(doc_count)

    rng = random.Random(seed)
    real_scores = []
    shuf_scores = []
    for text, toks, own in zip(target_sample, tok_lists, doc_sets):
        nb = len(toks) - 1
        if nb <= 0:
            continue
        # Real document: every bigram is in its own set, so it must occur in
        # at least two target documents to be in the leave-one-out set.
        hits = 0
        for b in zip(toks, toks[1:]):
            if doc_count[b] >= 2:
                hits += 1
        real_scores.append(hits / (nb + _BIGRAM_SHRINK))

        stoks = _shuffled_tokens(text, rng)
        snb = len(stoks) - 1
        if snb <= 0:
            continue
        hits = 0
        for b in zip(stoks, stoks[1:]):
            c = doc_count.get(b, 0)
            if c and (c - (1 if b in own else 0)) >= 1:
                hits += 1
        shuf_scores.append(hits / (snb + _BIGRAM_SHRINK))

    if not real_scores or not shuf_scores:
        return bigram_set, 0.0

    lo_real = float(np.percentile(np.asarray(real_scores), 5))
    hi_shuf = float(np.percentile(np.asarray(shuf_scores), 95))
    return bigram_set, 0.5 * (lo_real + hi_shuf)


def _order_stats(docs, bigram_set):
    """Per document: bigram_hit and the fraction of punctuation-pair bigrams."""
    n = len(docs)
    bigram_hit = np.zeros(n, dtype=np.float64)
    punct_frac = np.zeros(n, dtype=np.float64)
    contains = bigram_set.__contains__
    is_punct = _PUNCT_TOKENS.__contains__
    for i, d in enumerate(docs):
        toks = d.lower().split()
        nb = len(toks) - 1
        if nb <= 0:
            continue
        hits = sum(map(contains, zip(toks, toks[1:])))
        bigram_hit[i] = hits / (nb + _BIGRAM_SHRINK)
        flags = list(map(is_punct, toks))
        if any(flags):
            pairs = sum(1 for a, b in zip(flags, flags[1:]) if a and b)
            punct_frac[i] = pairs / nb
    return bigram_hit, punct_frac


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    counts = _token_counts(docs)
    scores = _overlap_scores(docs, target_sample)

    # Word-order score and its label-free threshold.
    bigram_set, threshold = _calibrate(target_sample, seed)
    bigram_hit, punct_frac = _order_stats(docs, bigram_set)
    if bigram_set:
        order_ok = bigram_hit >= threshold
        final = scores * bigram_hit
    else:
        # No target bigrams at all: the word-order score carries no information.
        order_ok = np.ones(n, dtype=bool)
        final = scores
    punct_ok = punct_frac <= _PUNCT_PAIR_MAX

    # Exact duplicates (after whitespace/case normalisation): keep the first copy.
    seen = set()
    unique = np.ones(n, dtype=bool)
    for i, d in enumerate(docs):
        key = hash(_WS.sub(" ", d).strip().lower())
        if key in seen:
            unique[i] = False
        else:
            seen.add(key)

    base = unique & (counts > 0)
    usable = base & order_ok & punct_ok

    if not usable.any():
        # Last resort so the trainer never gets an empty list: nothing passed
        # the word-order threshold, so fall back to the ranking alone.
        usable = base & punct_ok
        if not usable.any():
            usable = base
        if not usable.any():
            return list(range(n))

    order = np.argsort(-final, kind="stable")

    # Fill to T tokens with usable documents only.  If they hold fewer than T
    # tokens the list is returned as is and the trainer cycles it.
    selected = []
    total = 0
    for i in order:
        if not usable[i]:
            continue
        selected.append(int(i))
        total += int(counts[i])
        if total >= T:
            break

    random.Random(seed).shuffle(selected)
    return selected
