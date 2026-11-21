"""Curator: keep only clean, in-domain (WikiText-style) documents, deduplicated.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. exact dedupe (normalised text hash, plus first 300 normalised characters)
  2. junk filter (length, unique-line ratio, zlib ratio, letter/space fraction)
  3. domain filter (punctuation density check)
  4. shuffled-word filter (bigram hit rate against the target sample)
  5. seeded shuffle; extra reshuffled passes of the kept set if it is below T tokens
"""
import hashlib
import os
import random
import re
import string
import time
import zlib
import numpy as np

from common import T

MIN_CHARS = 200
MIN_UNIQUE_LINE_RATIO = 0.5
MIN_ZLIB_RATIO = 0.2
MIN_ALPHA_SPACE_FRAC = 0.8
PREFIX_CHARS = 300
BIGRAM_WORDS = 2000
THRESHOLD_SCALE = 0.85
THRESHOLD_PERCENTILE = 2
TOKENISE_DEADLINE_S = 110.0  # after this much elapsed time, estimate as chars/4

_ALPHA_SPACE_DELETE = {ord(c): None for c in string.ascii_letters + " "}

def _normalise(text):
    return " ".join(text.lower().split())

def _is_junk(text):
    n = len(text)
    if n < MIN_CHARS:
        return True

    alpha_space = n - len(text.translate(_ALPHA_SPACE_DELETE))
    if alpha_space < MIN_ALPHA_SPACE_FRAC * n:
        return True

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return True
    if len(set(lines)) / len(lines) < MIN_UNIQUE_LINE_RATIO:
        return True

    raw = text.encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw)) / len(raw) < MIN_ZLIB_RATIO:
        return True

    return False

def _in_domain(text):
    # WikiText style often has high punctuation-to-word ratio.
    # We count punctuation marks and words to ensure a structured, non-spammy flow.
    words = re.findall(r'\w+', text.lower())
    if not words:
        return False
    
    punct_count = 0
    for char in text:
        if char in string.punctuation:
            punct_count += 1
    
    # Targeting the punctuation density characteristic of high-quality encyclopedia text.
    # Usually, well-formatted articles have a significant ratio of punctuation to word count.
    ratio = punct_count / len(words)
    return 0.3 < ratio < 2.5

def _bigrams(text):
    words = text.lower().split()[:BIGRAM_WORDS]
    return list(zip(words, words[1:]))

def _hit_rate(text, bigram_set):
    bg = _bigrams(text)
    if not bg:
        return 0.0
    hits = 0
    for b in bg:
        if b in bigram_set:
            hits += 1
    return hits / len(bg)

def _bigram_threshold(target_sample):
    target_sample = list(target_sample)
    if len(target_sample) < 2:
        return None, None
    half = min(100, len(target_sample) // 2)
    build, calib = target_sample[:half], target_sample[half:half + 100]
    bigram_set = set()
    for d in build:
        bigram_set.update(_bigrams(d))
    if not bigram_set or not calib:
        return None, None
    rates = [_hit_rate(d, bigram_set) for d in calib]
    threshold = THRESHOLD_SCALE * float(np.percentile(rates, THRESHOLD_PERCENTILE))
    return bigram_set, threshold

def _count_tokens(docs, indices, t_start):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        return sum(len(docs[i]) for i in indices) / 4.0

    total = 0.0
    batch_size = 256
    threads = max(1, min(8, os.cpu_count() or 1))
    pos = 0
    while pos < len(indices):
        if time.time() - t_start > TOKENISE_DEADLINE_S:
            total += sum(len(docs[i]) for i in indices[pos:]) / 4.0
            break
        batch = [docs[i] for i in indices[pos:pos + batch_size]]
        try:
            encoded = enc.encode_ordinary_batch(batch, num_threads=threads)
            total += sum(len(e) for e in encoded)
        except Exception:
            total += sum(len(d) for d in batch) / 4.0
        pos += batch_size
    return total

def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    seen_hashes = set()
    seen_prefixes = set()
    deduped = []
    for i, text in enumerate(docs):
        if not isinstance(text, str):
            continue
        norm = _normalise(text)
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen_hashes:
            continue
        prefix = norm[:PREFIX_CHARS]
        if prefix in seen_prefixes:
            continue
        seen_hashes.add(h)
        seen_prefixes.add(prefix)
        deduped.append(i)
    del seen_hashes, seen_prefixes

    candidates = []
    for i in deduped:
        text = docs[i]
        if _is_junk(text):
            continue
        if not _in_domain(text):
            continue
        candidates.append(i)

    bigram_set, threshold = _bigram_threshold(target_sample)
    if bigram_set is None:
        kept = candidates
    else:
        kept = [i for i in candidates if _hit_rate(docs[i], bigram_set) >= threshold]

    if not kept:
        kept = candidates or deduped or list(range(len(docs)))

    kept = list(kept)
    rng.shuffle(kept)
    out = list(kept)

    pass_tokens = _count_tokens(docs, kept, t_start)
    if pass_tokens > 0:
        acc = pass_tokens
        while acc < T:
            extra = list(kept)
            rng.shuffle(extra)
            out.extend(extra)
            acc += pass_tokens

    return out
