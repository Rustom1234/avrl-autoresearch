"""Curator: keep clean, intact, in-domain (WikiText-style) documents, de-duplicated,
ranked by a length-normalized bigram score, in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
--------
1. Guards (hard): minimum length, non-ASCII letters, spam patterns, repetition.
2. Target similarity (hard):
   - length-normalized bigram score over the WHOLE document, with the first
     MAX_WORDS words weighted more heavily (HEAD_WEIGHT) than the remainder;
   - vocabulary overlap: Jaccard(set of doc words, set of target words) must be
     at least MIN_VOCAB_JACCARD.
3. Tiers:
   - Tier 1: passes everything, including the header rule;
   - Tier 2: passes everything except the header rule.
4. De-duplicate in priority order (Tier 1 first, higher score first).
5. Fill the budget T with Tier 1 (score descending), then Tier 2 (score
   descending); finally shuffle the selected documents with the seed.
"""
import os
import random
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_WORDS = 1500          # "head" of the document, prioritized in the bigram score
HEAD_WEIGHT = 0.7         # weight of the head score when the doc has a tail
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
MIN_VOCAB_JACCARD = 0.01  # minimum Jaccard(doc words, target words)
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")

BUDGET_MARGIN_TIKTOKEN = 1.02   # small safety margin over T when counting exactly
BUDGET_MARGIN_APPROX = 1.20     # larger margin when using chars/4
TOKENIZE_DEADLINE = 120.0       # seconds; after this fall back to chars/4
TOKENIZE_CHUNK = 2000


def _words(text):
    return text.lower().split()


def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)
        B.update(zip(w, w[1:]))
    return B


def _score_words(words, B):
    """Length-normalized bigram score over the whole document.

    head = fraction of bigrams within the first MAX_WORDS words that are in B
    tail = fraction of the remaining bigrams that are in B
    score = head                                   (no tail)
          = HEAD_WEIGHT*head + (1-HEAD_WEIGHT)*tail (with tail)
    """
    n = len(words) - 1
    if n <= 0:
        return 0.0
    contains = B.__contains__
    n_head = min(n, MAX_WORDS)
    head_words = words[: n_head + 1]
    head_hits = sum(map(contains, zip(head_words, head_words[1:])))
    head = head_hits / n_head
    n_tail = n - n_head
    if n_tail <= 0:
        return head
    tail_hits = sum(map(contains, zip(words[n_head:], words[n_head + 1:])))
    tail = tail_hits / n_tail
    return HEAD_WEIGHT * head + (1.0 - HEAD_WEIGHT) * tail


def _calibrate_tau(target_sample):
    """tau = 0.8 x 5th percentile of two-fold held-out target scores."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half = n // 2
    a, b = target_sample[:half], target_sample[half:]
    scores = []
    for build, held in ((a, b), (b, a)):
        B = _bigram_set(build)
        for t in held:
            w = _words(t)
            if len(w) >= 2:
                scores.append(_score_words(w, B))
    if not scores:
        return 0.0
    return 0.8 * float(np.percentile(np.asarray(scores), 5))


def _jaccard(word_set, vocab):
    if not word_set or not vocab:
        return 0.0
    inter = len(word_set & vocab)
    union = len(word_set) + len(vocab) - inter
    return inter / union if union else 0.0


def _calibrate_jaccard(target_sample, vocab):
    """Minimum vocabulary overlap.

    The requested threshold is MIN_VOCAB_JACCARD (0.01).  Jaccard against a large
    target vocabulary is bounded by |doc vocab| / |target vocab|, so if the target
    documents themselves would not clear 0.01 (large vocabulary, short articles)
    the threshold is lowered to half the 5th percentile of the target's own
    overlap, so the rule can never wipe out the in-domain documents.
    """
    if not vocab:
        return 0.0
    vals = []
    for t in target_sample:
        ws = set(_words(t))
        if ws:
            vals.append(_jaccard(ws, vocab))
    if not vals:
        return MIN_VOCAB_JACCARD
    p5 = float(np.percentile(np.asarray(vals), 5))
    return min(MIN_VOCAB_JACCARD, 0.5 * p5)


def _nonascii_letter_frac(text):
    n = len(text)
    if n == 0:
        return 0.0
    # Fast pre-check: total non-ASCII characters.
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii <= MAX_NONASCII_LETTER_FRAC * n:
        return 0.0
    cnt = sum(1 for c in text if ord(c) > 127 and c.isalpha())
    return cnt / n


def _repetitive(words):
    n = len(words) - 4
    if n <= 0:
        return False
    top = Counter(
        zip(words, words[1:], words[2:], words[3:], words[4:])
    ).most_common(1)[0][1]
    # require top >= 3 so that very short docs are not dropped trivially
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n


def _fill_budget(docs, ordered, t0):
    """Take documents from `ordered` (priority order) until the budget is filled.

    Returns (selected, token_estimate, kind).
    """
    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0.0
    exact = enc is not None
    pos = 0
    n = len(ordered)
    while pos < n:
        use_enc = enc is not None and (time.time() - t0) < TOKENIZE_DEADLINE
        chunk = ordered[pos: pos + TOKENIZE_CHUNK]
        pos += len(chunk)
        counts = None
        if use_enc:
            try:
                counts = [
                    len(x)
                    for x in enc.encode_ordinary_batch([docs[i] for i in chunk])
                ]
            except Exception:
                counts = None
                enc = None
        if counts is None:
            exact = False
            counts = [len(docs[i]) / 4.0 for i in chunk]
        margin = BUDGET_MARGIN_TIKTOKEN if exact else BUDGET_MARGIN_APPROX
        budget = margin * T
        done = False
        for i, c in zip(chunk, counts):
            selected.append(i)
            total += c
            if total >= budget:
                done = True
                break
        if done:
            break
    return selected, int(total), ("tiktoken" if exact else "approx")


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    vocab = set()
    for t in target_sample:
        vocab.update(_words(t))
    min_jacc = _calibrate_jaccard(target_sample, vocab)

    # Only enforce the header rule if the target sample really follows it.
    if target_sample:
        hdr_frac = sum(
            1 for t in target_sample if t.lstrip().startswith("= ")
        ) / len(target_sample)
    else:
        hdr_frac = 0.0
    require_header = hdr_frac >= 0.9

    # candidates: (tier, -score, index, full_hash, prefix)
    candidates = []
    reasons = Counter()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            reasons["short"] += 1
            continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC:
            reasons["non_ascii"] += 1
            continue

        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS):
            reasons["spam"] += 1
            continue

        words = low.split()
        score = _score_words(words, B)
        if score < tau:
            reasons["low_score"] += 1
            continue
        if vocab and _jaccard(set(words), vocab) < min_jacc:
            reasons["low_vocab_overlap"] += 1
            continue
        if _repetitive(words):
            reasons["repetition"] += 1
            continue

        has_header = text.lstrip().startswith("= ")
        tier = 1 if (has_header or not require_header) else 2

        norm = " ".join(words)
        candidates.append((tier, -score, i, hash(norm), norm[:300]))

    # Priority order: Tier 1 by score descending, then Tier 2 by score descending.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    # Dedup in priority order, so the best copy (Tier 1, highest score) wins.
    seen_full = set()
    seen_prefix = set()
    ordered = []
    n_tier = Counter()
    for tier, _neg, i, h, prefix in candidates:
        if h in seen_full or prefix in seen_prefix:
            reasons["duplicate"] += 1
            continue
        seen_full.add(h)
        seen_prefix.add(prefix)
        ordered.append(i)
        n_tier[tier] += 1
    del candidates, seen_full, seen_prefix

    if not ordered:
        # Degenerate case: never return an empty list.
        print("[curate] WARNING: filter kept nothing; falling back to raw shuffle")
        kept = list(range(len(docs)))
        random.Random(seed).shuffle(kept)
        return kept

    selected, est_tokens, tok_kind = _fill_budget(docs, ordered, t0)
    n_t1 = n_tier[1]
    sel_t1 = min(len(selected), n_t1)
    sel_t2 = len(selected) - sel_t1

    random.Random(seed).shuffle(selected)

    print(
        "[curate] pool=%d eligible=%d (tier1=%d tier2=%d) selected=%d "
        "(tier1=%d tier2=%d) tokens~%d (%s, %.2fx T) tau=%.4f min_jaccard=%.5f "
        "header_rule=%s dropped=%s time=%.1fs"
        % (
            len(docs),
            len(ordered),
            n_tier[1],
            n_tier[2],
            len(selected),
            sel_t1,
            sel_t2,
            est_tokens,
            tok_kind,
            est_tokens / float(T),
            tau,
            min_jacc,
            require_header,
            dict(reasons),
            time.time() - t0,
        )
    )
    return selected
