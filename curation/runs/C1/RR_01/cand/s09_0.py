"""Curator: target-normalised bigram/unigram overlap score, exact dedup, cut at ~1.02*T tokens, seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Scoring: overlaps are measured on the first WINDOW whitespace tokens of each doc
(the 'lead'), and each is divided by the average overlap that target documents
themselves reach (leave-one-out, so a target doc is not compared with itself):

    score = b / b_target + 0.5 * (u / u_target)
"""
import os
import time
import random
import hashlib
from collections import Counter

from common import T

WINDOW = 1000             # score each doc on its first N whitespace tokens
MIN_TOKENS = 50           # drop docs shorter than this (whitespace tokens)
BUDGET_FACTOR = 1.02      # keep about 1.02 * T GPT-2 tokens
TIME_BUDGET_EXACT = 110.0 # seconds after which token counting falls back to len/4
BATCH = 512
UNIGRAM_WEIGHT = 0.5
TITLE_BONUS = 0.1


def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")


def _target_baselines(target_tokens, uni_df, bi_df):
    """Average leave-one-out unigram and bigram overlap of the target docs.

    A token (bigram) of a target doc counts as overlapping when it also occurs
    in at least one *other* target doc, i.e. its document frequency is >= 2.
    Overlaps are measured on the first WINDOW tokens, as for pool docs.
    """
    u_vals = []
    b_vals = []
    for toks in target_tokens:
        head = toks[:WINDOW]
        n = len(head)
        if n == 0:
            continue
        u_vals.append(sum(1 for t in head if uni_df[t] >= 2) / n)
        if n > 1:
            b_vals.append(
                sum(1 for bg in zip(head, head[1:]) if bi_df[bg] >= 2) / (n - 1)
            )
    u_target = sum(u_vals) / len(u_vals) if u_vals else 0.0
    b_target = sum(b_vals) / len(b_vals) if b_vals else 0.0
    if u_target <= 1e-9:
        u_target = 1.0
    if b_target <= 1e-9:
        b_target = 1.0
    return b_target, u_target


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Target unigram vocabulary and adjacent-bigram set (case preserved),
    # with document frequencies for the leave-one-out baselines.
    target_tokens = [doc.split() for doc in target_sample]
    uni_df = Counter()
    bi_df = Counter()
    for toks in target_tokens:
        uni_df.update(set(toks))
        bi_df.update(set(zip(toks, toks[1:])))
    vocab = set(uni_df)
    bigrams = set(bi_df)

    b_target, u_target = _target_baselines(target_tokens, uni_df, bi_df)
    del uni_df, bi_df

    # Score, filter and deduplicate the pool.
    seen = set()
    scored = []
    for i, text in enumerate(docs):
        toks = text.split()
        if len(toks) < MIN_TOKENS:
            continue

        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        head = toks[:WINDOW]
        n = len(head)
        u = sum(1 for t in head if t in vocab) / n
        if n > 1:
            b = sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1)
        else:
            b = 0.0

        score = b / b_target + UNIGRAM_WEIGHT * (u / u_target)
        if _has_title_line(text):
            score += TITLE_BONUS
        scored.append((score, i))
    del seen

    if not scored:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Best first; index as a deterministic tie-break.
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

    # Take from the top until about 1.02*T GPT-2 tokens.
    budget = BUDGET_FACTOR * T
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0.0
    pos = 0
    done = False
    while pos < len(ranked) and not done:
        batch_idx = ranked[pos:pos + BATCH]
        pos += BATCH

        lengths = None
        if enc is not None and (time.time() - t_start) < TIME_BUDGET_EXACT:
            try:
                encoded = enc.encode_ordinary_batch([docs[i] for i in batch_idx])
                lengths = [len(e) for e in encoded]
            except Exception:
                lengths = None
        if lengths is None:
            lengths = [len(docs[i]) / 4.0 for i in batch_idx]

        for i, n_tok in zip(batch_idx, lengths):
            selected.append(i)
            total += n_tok
            if total >= budget:
                done = True
                break

    # Uniform mix rather than best-to-worst order.
    rng.shuffle(selected)
    return selected
