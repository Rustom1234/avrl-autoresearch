"""Curator: overlap x coherence quality score, exact dedup, spam filter, seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Scoring:
  overlap   = bigram-overlap + 0.25 * unigram-overlap with the target sample
              (a cheap surrogate for the log-probability under the target), plus
              a small bonus for a "= Title =" first line.
  coherence = distinct bigrams / total bigrams in the first 1000 tokens
              (low for repetitive spam, close to 1 for natural text).
  quality   = overlap * coherence.

Selection:
  Tier 1: non-spam docs with high coherence and high overlap (the overlap
          threshold is calibrated on held-out target docs), best quality first,
          up to ~1.02*T tokens.
  Fill:   only if tier 1 holds less than 0.8*T tokens, the rest of the budget is
          filled with the remaining non-spam docs by raw bigram overlap,
          regardless of coherence.
"""
import os
import time
import random
import hashlib

from common import T

MAX_SCORE_TOKENS = 2000    # overlap is scored on the first N whitespace tokens
COHERENCE_TOKENS = 1000    # coherence is measured on the first N whitespace tokens
MIN_TOKENS = 50            # drop docs shorter than this (whitespace tokens)
BUDGET_FACTOR = 1.02       # keep about 1.02 * T GPT-2 tokens
FILL_TRIGGER = 0.8         # fill only if tier 1 holds less than this * T tokens
COHERENCE_MIN = 0.7        # "high coherence" threshold
OVERLAP_FRAC = 0.6         # "high overlap" = this * low percentile of held-out target
OVERLAP_PCT = 0.10         # the low percentile used for calibration
HOLDOUT_EVERY = 4          # every 4th target doc is held out for calibration
TITLE_BONUS = 0.1
TIME_BUDGET_EXACT = 110.0  # seconds after which token counting falls back to len/4
BATCH = 512


def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")


def _overlap(head, vocab, bigrams):
    n = len(head)
    if n == 0:
        return 0.0, 0.0
    u = sum(1 for t in head if t in vocab) / n
    if n > 1:
        b = sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1)
    else:
        b = 0.0
    return u, b


def _coherence(toks):
    head = toks[:COHERENCE_TOKENS]
    n = len(head) - 1
    if n <= 0:
        return 0.0
    return len(set(zip(head, head[1:]))) / n


def _is_spam(text, n_toks):
    excl = text.count("!")
    if excl > max(5, 0.02 * n_toks):
        return True
    urls = text.count("http://") + text.count("https://") + text.count("www.")
    if urls > max(3, 0.005 * n_toks):
        return True
    return False


def _calibrate_overlap_threshold(target_sample):
    """Low percentile of the overlap score of held-out target docs, scaled down."""
    train_vocab = set()
    train_bigrams = set()
    held = []
    for j, doc in enumerate(target_sample):
        toks = doc.split()
        if j % HOLDOUT_EVERY == 0:
            held.append(toks)
        else:
            train_vocab.update(toks)
            train_bigrams.update(zip(toks, toks[1:]))
    scores = []
    for toks in held:
        head = toks[:MAX_SCORE_TOKENS]
        if len(head) < 2:
            continue
        u, b = _overlap(head, train_vocab, train_bigrams)
        scores.append(b + 0.25 * u)
    if not scores:
        return 0.0
    scores.sort()
    return OVERLAP_FRAC * scores[int(OVERLAP_PCT * (len(scores) - 1))]


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Target unigram vocabulary and adjacent-bigram set (case preserved).
    vocab = set()
    bigrams = set()
    for doc in target_sample:
        toks = doc.split()
        vocab.update(toks)
        bigrams.update(zip(toks, toks[1:]))

    overlap_min = _calibrate_overlap_threshold(target_sample)

    # Score, filter and deduplicate the pool.
    seen = set()
    tier1 = []      # (quality, index)
    rest = []       # (bigram overlap, index): non-spam docs outside tier 1
    for i, text in enumerate(docs):
        toks = text.split()
        if len(toks) < MIN_TOKENS:
            continue

        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        if _is_spam(text, len(toks)):
            continue

        u, b = _overlap(toks[:MAX_SCORE_TOKENS], vocab, bigrams)
        base = b + 0.25 * u
        coh = _coherence(toks)

        if coh >= COHERENCE_MIN and base >= overlap_min:
            overlap = base + (TITLE_BONUS if _has_title_line(text) else 0.0)
            tier1.append((overlap * coh, i))
        else:
            rest.append((b, i))
    del seen

    if not tier1 and not rest:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Best first; index as a deterministic tie-break.
    tier1.sort(key=lambda x: (-x[0], x[1]))
    rest.sort(key=lambda x: (-x[0], x[1]))
    tier1_ranked = [i for _, i in tier1]
    rest_ranked = [i for _, i in rest]

    budget = BUDGET_FACTOR * T
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0.0

    def take(ranked):
        """Append docs from ranked until the budget is reached."""
        nonlocal total
        pos = 0
        while pos < len(ranked) and total < budget:
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
                    break

    # Tier 1: high-coherence, high-overlap docs by quality.
    take(tier1_ranked)

    # Fill by raw bigram overlap only if tier 1 is clearly too small.
    if total < FILL_TRIGGER * T:
        take(rest_ranked)

    if not selected:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Uniform mix rather than best-to-worst order.
    rng.shuffle(selected)
    return selected
