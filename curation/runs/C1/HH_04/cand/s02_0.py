"""Curator: junk/spam filter, MinHash near-duplicate removal, then seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re, zlib
import numpy as np

NUM_PERM = 64          # MinHash permutations
BANDS = 16             # LSH bands
ROWS = 4               # rows per band (BANDS * ROWS == NUM_PERM)
SHINGLE = 5            # word n-gram size
JACCARD_MAX = 0.5      # keep a doc only if estimated Jaccard with every kept doc is below this
MIN_ALPHA = 0.7        # minimum alphabetic-character fraction
PRIME = 2147483647     # 2^31 - 1, modulus for universal hashing

SPAM_RE = re.compile(
    r"click here|coupon|discount code|free shipping|\border\b[^.\n]{0,40}\bonline\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\w+")
LETTER_RE = re.compile(r"[^\W\d_]")


def _alpha_ok(text):
    n = len(text)
    if n == 0:
        return False
    return len(LETTER_RE.findall(text)) / n >= MIN_ALPHA


def _shingle_hashes(text):
    words = WORD_RE.findall(text.lower())
    if not words:
        return None
    if len(words) < SHINGLE:
        grams = {" ".join(words)}
    else:
        grams = {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}
    # crc32 is deterministic across processes (unlike str hash()).
    return np.fromiter(
        (zlib.crc32(g.encode("utf-8")) % PRIME for g in grams),
        dtype=np.int64,
        count=len(grams),
    )


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # Seeded shuffle: this order is the final output order and also decides
    # which member of a near-duplicate cluster is kept.
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)

    # Stage 1: drop junk and spam.
    cand = []
    for i in idx:
        t = docs[i]
        if not isinstance(t, str) or not _alpha_ok(t) or SPAM_RE.search(t):
            continue
        cand.append(i)

    # Stage 2: MinHash signatures with seeded universal hash permutations.
    rng = np.random.default_rng(seed)
    A = rng.integers(1, PRIME, size=NUM_PERM, dtype=np.int64)
    B = rng.integers(0, PRIME, size=NUM_PERM, dtype=np.int64)

    def signature(h):
        vals = (A[:, None] * h[None, :] + B[:, None]) % PRIME
        return vals.min(axis=1)

    # Stage 3: greedy near-duplicate removal in shuffled order.
    # LSH banding gives candidate kept docs; the exact-per-permutation Jaccard
    # estimate (fraction of matching signature positions) decides duplication.
    buckets = [dict() for _ in range(BANDS)]
    sigs = {}
    kept = []
    for i in cand:
        h = _shingle_hashes(docs[i])
        if h is None:
            continue
        sig = signature(h)
        keys = [sig[b * ROWS:(b + 1) * ROWS].tobytes() for b in range(BANDS)]

        cands = set()
        for b, k in enumerate(keys):
            cands.update(buckets[b].get(k, ()))

        if any(np.mean(sigs[j] == sig) >= JACCARD_MAX for j in cands):
            continue

        kept.append(i)
        sigs[i] = sig
        for b, k in enumerate(keys):
            buckets[b].setdefault(k, []).append(i)

    # Survivors are already in seeded shuffle order.
    return kept
