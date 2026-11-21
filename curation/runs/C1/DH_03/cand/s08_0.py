"""Curator: keep clean, target-like documents, deduplicate, emit shuffled passes.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Bigram fluency score against the target sample, plus junk guards.
  2. Global threshold: 5th percentile of held-out target scores times 0.95.
  3. Article-format docs (first line "= Title =") get a valley threshold from
     1-D 2-means on their scores, clamped to [0.6 * threshold, threshold].
  4. Dedup in descending score, so the most fluent copy of a duplicate wins.
  5. Clean-only output: exact tiktoken counts, a 0.08 T safety floor only, and a
     fresh shuffle per pass until T plus one pass of margin is covered.
"""
import os
import random
import re
import hashlib
from collections import Counter

import numpy as np

from common import T

TOP_BIGRAMS = 5000
MIN_CHARS = 200
PREFIX_CHARS = 300
FLOOR_FRACTION = 0.08
KMEANS_ITERS = 20
MIN_CENTRE_GAP = 0.05
ARTICLE_THR_FLOOR = 0.6
TOKEN_BATCH = 2000
FILL_BATCH = 256

_ODD_RE = re.compile(r"[^A-Za-z0-9\s.,;:!?'\"()\-]")
_HEADER_RE = re.compile(r"^= [^=].* =$")


def _bigram_set(texts, k=TOP_BIGRAMS):
    counts = Counter()
    for t in texts:
        w = t.split()
        counts.update(zip(w, w[1:]))
    return {b for b, _ in counts.most_common(k)}


def _score(text, bigrams):
    w = text.split()
    n = len(w) - 1
    if n <= 0:
        return 0.0
    return sum(map(bigrams.__contains__, zip(w, w[1:]))) / n


def _nonascii_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (n - len(text.encode("ascii", "ignore"))) / n


def _odd_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return len(_ODD_RE.findall(text)) / n


def _hashes(text):
    norm = " ".join(text.lower().split())
    full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
    prefix = hashlib.md5(norm[:PREFIX_CHARS].encode("utf-8", "ignore")).digest()
    return full, prefix


def _is_article(text):
    # First line in the target title format "= Title =". Leading blank lines and
    # surrounding whitespace on the line are ignored.
    head = text.lstrip()
    if not head:
        return False
    first = head.split("\n", 1)[0].strip()
    return _HEADER_RE.match(first) is not None


def _valley_threshold(values, threshold):
    """Article threshold from 1-D 2-means on article-format scores."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return threshold
    lo = float(np.percentile(x, 10))
    hi = float(np.percentile(x, 90))
    for _ in range(KMEANS_ITERS):
        mid = 0.5 * (lo + hi)
        high_mask = x >= mid
        if high_mask.any():
            hi = float(x[high_mask].mean())
        if (~high_mask).any():
            lo = float(x[~high_mask].mean())
    if hi - lo < MIN_CENTRE_GAP:
        return threshold
    thr_valley = 0.5 * (lo + hi)
    return min(threshold, max(thr_valley, ARTICLE_THR_FLOOR * threshold))


class _TokenCounter:
    """Exact GPT-2 token counts (falls back to chars / 4 without tiktoken)."""

    def __init__(self):
        self.enc = None
        try:
            import tiktoken

            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None
        self.threads = max(1, min(8, os.cpu_count() or 1))

    def count(self, texts):
        if not texts:
            return []
        if self.enc is None:
            return [max(1, len(t) // 4) for t in texts]
        out = []
        for s in range(0, len(texts), TOKEN_BATCH):
            chunk = texts[s:s + TOKEN_BATCH]
            try:
                toks = self.enc.encode_ordinary_batch(chunk, num_threads=self.threads)
                out.extend(len(t) for t in toks)
            except Exception:
                for t in chunk:
                    try:
                        out.append(len(self.enc.encode_ordinary(t)))
                    except Exception:
                        out.append(max(1, len(t) // 4))
        return out


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    target = list(target_sample)
    n = len(docs)

    # 1. Bigram fluency: held-out scores for the target sample.
    half = len(target) // 2
    part_a, part_b = target[:half], target[half:]
    set_a = _bigram_set(part_a)
    set_b = _bigram_set(part_b)
    heldout = [_score(t, set_a) for t in part_b] + [_score(t, set_b) for t in part_a]

    final_set = _bigram_set(target)

    # 2. Global threshold.
    threshold = float(np.percentile(heldout, 5)) * 0.95 if heldout else 0.0

    # 3. Junk guard limits from the target sample.
    max_nonascii = max((_nonascii_share(t) for t in target), default=0.0) * 1.5
    max_odd = max((_odd_share(t) for t in target), default=0.0) * 1.5

    scores = np.zeros(n, dtype=np.float64)
    passed_guard = np.zeros(n, dtype=bool)
    is_article = np.zeros(n, dtype=bool)

    for i, d in enumerate(docs):
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_share(d) > max_nonascii:
            continue
        if _odd_share(d) > max_odd:
            continue
        passed_guard[i] = True
        scores[i] = _score(d, final_set)
        is_article[i] = _is_article(d)

    # 4. Valley threshold for article-format docs.
    article_scores = scores[passed_guard & is_article]
    thr_article = _valley_threshold(article_scores, threshold)

    # 5. Selection.
    candidates = []
    rejected = []
    for i in range(n):
        if not passed_guard[i]:
            continue
        thr = thr_article if is_article[i] else threshold
        if scores[i] >= thr:
            candidates.append(i)
        else:
            rejected.append(i)

    # 6. Dedup in descending score: the most fluent copy wins.
    candidates.sort(key=lambda i: (-scores[i], i))
    seen_full = set()
    seen_prefix = set()
    kept = []
    for i in candidates:
        full, prefix = _hashes(docs[i])
        if full in seen_full or prefix in seen_prefix:
            continue
        seen_full.add(full)
        seen_prefix.add(prefix)
        kept.append(i)

    # 7. Exact token counts for the clean set.
    counter = _TokenCounter()
    kept_counts = counter.count([docs[i] for i in kept])
    pass_tokens = int(sum(kept_counts))

    # 8. Safety floor only: top up with the best rejected docs if the clean set
    #    is smaller than 0.08 T tokens.
    floor_tokens = FLOOR_FRACTION * T
    if pass_tokens < floor_tokens and rejected:
        rejected.sort(key=lambda i: (-scores[i], i))
        pos = 0
        while pass_tokens < floor_tokens and pos < len(rejected):
            batch = []
            while pos < len(rejected) and len(batch) < FILL_BATCH:
                i = rejected[pos]
                pos += 1
                full, prefix = _hashes(docs[i])
                if full in seen_full or prefix in seen_prefix:
                    continue
                seen_full.add(full)
                seen_prefix.add(prefix)
                batch.append(i)
            if not batch:
                continue
            for i, c in zip(batch, counter.count([docs[i] for i in batch])):
                if pass_tokens >= floor_tokens:
                    break
                kept.append(i)
                pass_tokens += int(c)

    if not kept:
        kept = list(range(n))
        pass_tokens = int(sum(counter.count(list(docs))))

    kept.sort()

    # 9. Multi-pass output: a fresh shuffle per pass until T plus one pass of
    #    margin is covered.
    if pass_tokens <= 0:
        order = list(kept)
        rng.shuffle(order)
        return order

    out = []
    total = 0
    target_tokens = T + pass_tokens
    while total < target_tokens:
        order = list(kept)
        rng.shuffle(order)
        out.extend(order)
        total += pass_tokens
    return out
