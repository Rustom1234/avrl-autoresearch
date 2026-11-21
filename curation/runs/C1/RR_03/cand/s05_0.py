"""Curator: fill the token budget T with the best documents, tier by tier.

Every tier passes the hard junk rules (length, non-ASCII, spam, blob junk,
repetition) and is de-duplicated. The tiers differ in the remaining rules:

  Tier 1: header rule and bigram score >= tau
  Tier 2: bigram score >= tau, header rule waived
  Tier 3: bigram score and header rules both waived

Tiers are used in order. Inside a tier, documents are sorted by bigram score
(descending) and picked until the budget T is full. The final selection is
shuffled with the seed from the environment variable CURATE_SEED.

select(docs, target_sample) -> ordered list of indices into docs.
"""
import os
import random
import string
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_WORDS = 1500          # words used for the bigram score
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")

# Blob-junk limits (floors; raised if the target sample itself is above them).
MAX_PUNCT_FRAC = 0.15
MAX_MEAN_WORD_LEN = 12.0
TOKENIZE_DEADLINE = 110.0  # seconds; after this, token counts fall back to chars/4

_PUNCT_TABLE = {ord(c): None for c in string.punctuation}


def _words(text):
    return text.lower().split()


def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)[:MAX_WORDS + 1]
        B.update(zip(w, w[1:]))
    return B


def _score_words(words, B):
    """Fraction of the bigrams in the first MAX_WORDS words that are in B."""
    w = words[:MAX_WORDS]
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(w, w[1:]):
        if b in B:
            hits += 1
    return hits / n


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


def _punct_frac(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (n - len(text.translate(_PUNCT_TABLE))) / n


def _repetitive(words):
    n = len(words) - 4
    if n <= 0:
        return False
    top = Counter(
        zip(words, words[1:], words[2:], words[3:], words[4:])
    ).most_common(1)[0][1]
    # require top >= 3 so that very short docs are not dropped trivially
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n


def _calibrate_blob(target_sample):
    """Blob-junk limits: fixed floors, loosened if the target sample needs it."""
    max_punct = MAX_PUNCT_FRAC
    max_wlen = MAX_MEAN_WORD_LEN
    pf, wl = [], []
    for t in target_sample:
        if not t:
            continue
        pf.append(_punct_frac(t))
        nw = len(t.split())
        if nw:
            wl.append(len(t) / nw)
    if pf:
        max_punct = max(max_punct, 1.5 * float(np.percentile(np.asarray(pf), 99)))
    if wl:
        max_wlen = max(max_wlen, 1.5 * float(np.percentile(np.asarray(wl), 99)))
    return max_punct, max_wlen


def _token_counts(docs, idxs, t0, state):
    """Token counts for docs[idxs]; tiktoken if time allows, else chars/4."""
    if not idxs:
        return []
    if state.get("enc") is None and not state.get("failed"):
        try:
            import tiktoken

            state["enc"] = tiktoken.get_encoding("gpt2")
        except Exception:
            state["failed"] = True
    enc = state.get("enc")
    out = []
    CH = 2000
    for s in range(0, len(idxs), CH):
        chunk = idxs[s:s + CH]
        if enc is not None and time.time() - t0 < TOKENIZE_DEADLINE:
            try:
                out.extend(
                    len(x)
                    for x in enc.encode_ordinary_batch([docs[i] for i in chunk])
                )
                continue
            except Exception:
                enc = None
                state["enc"] = None
                state["failed"] = True
        state["approx"] = True
        out.extend(max(1, len(docs[i]) // 4) for i in chunk)
    return out


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)
    max_punct, max_wlen = _calibrate_blob(target_sample)

    # Only enforce the header rule if the target sample really follows it.
    if target_sample:
        hdr_frac = sum(
            1 for t in target_sample if t.lstrip().startswith("= ")
        ) / len(target_sample)
    else:
        hdr_frac = 0.0
    require_header = hdr_frac >= 0.9

    reasons = Counter()
    # tier -> list of (-score, index, full_hash, prefix)
    tiers = {1: [], 2: [], 3: []}

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            reasons["short"] += 1
            continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC:
            reasons["non_ascii"] += 1
            continue
        if _punct_frac(text) > max_punct:
            reasons["blob_punct"] += 1
            continue

        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS):
            reasons["spam"] += 1
            continue

        words = low.split()
        if not words or len(text) / len(words) > max_wlen:
            reasons["blob_words"] += 1
            continue
        if _repetitive(words):
            reasons["repetition"] += 1
            continue

        score = _score_words(words, B)
        has_header = text.lstrip().startswith("= ")
        if score >= tau:
            tier = 1 if (has_header or not require_header) else 2
        else:
            tier = 3

        norm = " ".join(words)
        tiers[tier].append((-score, i, hash(norm), norm[:300]))

    # Fill the budget tier by tier, best bigram score first, with dedup.
    seen_full = set()
    seen_prefix = set()
    selected = []
    total = 0
    tok_state = {}
    tier_stats = {}

    for tier in (1, 2, 3):
        cand = tiers[tier]
        if total >= T or not cand:
            tier_stats[tier] = (len(cand), 0, 0)
            continue
        cand.sort(key=lambda c: (c[0], c[1]))  # score descending, then index

        uniq = []
        for neg_s, i, h, prefix in cand:
            if h in seen_full or prefix in seen_prefix:
                reasons["duplicate"] += 1
                continue
            seen_full.add(h)
            seen_prefix.add(prefix)
            uniq.append(i)

        picked = 0
        picked_tok = 0
        CH = 4000
        for s in range(0, len(uniq), CH):
            if total >= T:
                break
            chunk = uniq[s:s + CH]
            counts = _token_counts(docs, chunk, t0, tok_state)
            for i, n in zip(chunk, counts):
                if total >= T:
                    break
                selected.append(i)
                total += n
                picked += 1
                picked_tok += n
        tier_stats[tier] = (len(cand), picked, picked_tok)

    if not selected:
        # Degenerate case: never return an empty list.
        print("[curate] WARNING: filter kept nothing; falling back to raw shuffle")
        selected = list(range(len(docs)))

    random.Random(seed).shuffle(selected)

    print(
        "[curate] pool=%d selected=%d tokens~%d (%s, %.2fx T) tau=%.4f "
        "header_rule=%s max_punct=%.3f max_wlen=%.1f "
        "tiers(cand,picked,tokens)=%s dropped=%s time=%.1fs"
        % (
            len(docs),
            len(selected),
            total,
            "partly chars/4" if tok_state.get("approx") else "tiktoken",
            total / float(T),
            tau,
            require_header,
            max_punct,
            max_wlen,
            tier_stats,
            dict(reasons),
            time.time() - t0,
        )
    )
    return selected
