"""Curator: format gate + damage filter + junk filter + dedupe, in-domain only.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. format gate   - WikiText-looking docs only (title line or tokenisation markers)
  2. damage filter - whitespace-word bigram hit fraction against the target sample,
                     computed per consecutive 300-word window over the WHOLE body
                     (a final window shorter than 100 words is merged into the
                     previous one); the document score is the MINIMUM over its
                     windows. Threshold = 0.8 * 5th percentile of the target docs'
                     leave-one-out minimum window scores.
  3. junk filter   - too short, too compressible, too much punctuation
  4. dedupe        - exact (normalised hash) and near (title + first 200 chars)
  5. order / fill  - seeded shuffle; if survivors < 0.5 T tokens, top up to T with
                     clean out-of-domain docs ranked by bigram hit fraction

Note on the fill step: out-of-domain docs are not WikiText-tokenised, so they are
scored (and checked for word-shuffling) with alphabetic-word bigrams instead of
whitespace-word bigrams, against a leave-one-out floor computed the same way
(first 600 words only, as before).
"""
import hashlib
import os
import random
import re
import time
import zlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_TITLE_RE = re.compile(r"^= .+ =$")
_MARKERS = (" @-@ ", " @,@ ", " @.@ ", " , ", " . ", " ( ")
_ALPHA_OR_SPACE_RE = re.compile(r"[^\W\d_]+|\s+")
_ALPHA_WORD_RE = re.compile(r"[a-z]+")

_SCORE_WORDS = 600           # fill-path scorer only
_SCORE_CHARS = 8000          # enough characters to cover 600 words
_WINDOW_WORDS = 300          # damage filter window size
_WINDOW_MIN_TAIL = 100       # shorter final windows are merged into the previous
_ZLIB_CAP = 200_000          # chars compressed per doc for the boilerplate check
_TIME_BUDGET = 150.0         # seconds; leave margin under the 180 s limit


# ----------------------------------------------------------------- helpers

def _split_title(text):
    """Return (title_line, body). Title is '' if the first line is not a heading."""
    s = text.lstrip()
    nl = s.find("\n")
    first = (s if nl < 0 else s[:nl]).strip()
    if _TITLE_RE.match(first):
        return first, ("" if nl < 0 else s[nl + 1:])
    return "", s


def _first_line(text):
    s = text.lstrip()
    nl = s.find("\n")
    return (s if nl < 0 else s[:nl]).strip()


def _format_ok(text):
    if _TITLE_RE.match(_first_line(text)):
        return True
    n = len(text)
    if n == 0:
        return False
    hits = 0
    for m in _MARKERS:
        hits += text.count(m)
    return hits * 1000.0 / n >= 3.0


def _is_junk(text):
    n = len(text)
    if n < 200:
        return True
    # more than 30% non-alphabetic, non-space characters
    non_alpha = len(_ALPHA_OR_SPACE_RE.sub("", text))
    if non_alpha > 0.30 * n:
        return True
    # repeated boilerplate compresses extremely well
    raw = text[:_ZLIB_CAP].encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw)) / len(raw) < 0.2:
        return True
    return False


def _tok_ws(s):
    return s.lower().split()


def _tok_alpha(s):
    return _ALPHA_WORD_RE.findall(s.lower())


def _build_scorer(target_bodies, tok):
    """Return (score_fn, leave_one_out_scores) for a given word tokeniser.

    Head-of-document scorer (first _SCORE_WORDS words); used by the fill path.
    """
    df = Counter()
    heads = []
    for body in target_bodies:
        w = tok(body)
        df.update(set(zip(w, w[1:])))
        heads.append(w[:_SCORE_WORDS])

    loo = []
    for w in heads:
        if len(w) < 2:
            continue
        # a bigram counts as a hit only if some *other* target doc has it
        hit = 0
        for b in zip(w, w[1:]):
            if df.get(b, 0) >= 2:
                hit += 1
        loo.append(hit / (len(w) - 1))

    known = set(df)

    def score(body):
        w = tok(body[:_SCORE_CHARS])[:_SCORE_WORDS]
        if len(w) < 2:
            return 0.0
        hit = 0
        for b in zip(w, w[1:]):
            if b in known:
                hit += 1
        return hit / (len(w) - 1)

    return score, loo


def _window_bounds(nw):
    """Consecutive [start, end) word windows; a short final window is merged."""
    bounds = [(s, min(s + _WINDOW_WORDS, nw)) for s in range(0, nw, _WINDOW_WORDS)]
    if len(bounds) >= 2 and bounds[-1][1] - bounds[-1][0] < _WINDOW_MIN_TAIL:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds


def _min_window_fraction(flags, nw):
    """Minimum per-window hit fraction.

    flags[j] is True iff the bigram (word j, word j+1) is a hit; len == nw - 1.
    A window [s, e) owns the bigrams starting at s .. e-2.
    """
    if nw < 2:
        return 0.0
    best = None
    for s, e in _window_bounds(nw):
        nb = e - s - 1
        if nb < 1:
            continue
        frac = sum(flags[s:e - 1]) / nb
        if best is None or frac < best:
            best = frac
    return 0.0 if best is None else best


def _build_window_scorer(target_bodies, tok):
    """Return (score_fn, leave_one_out_minima) for the whole-document damage filter."""
    df = Counter()
    words = []
    for body in target_bodies:
        w = tok(body)
        df.update(set(zip(w, w[1:])))
        words.append(w)

    loo = []
    for w in words:
        if len(w) < 2:
            continue
        # a bigram counts as a hit only if some *other* target doc has it
        flags = [df.get(b, 0) >= 2 for b in zip(w, w[1:])]
        loo.append(_min_window_fraction(flags, len(w)))

    known = set(df)

    def score(body):
        w = tok(body)
        if len(w) < 2:
            return 0.0
        flags = [b in known for b in zip(w, w[1:])]
        return _min_window_fraction(flags, len(w))

    return score, loo


def _dedupe_keys(text):
    norm = " ".join(text.lower().split())
    h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
    title = " ".join(_first_line(text).lower().split())
    return h, (title, norm[:200])


def _dedupe(indices, docs, seen_hashes, seen_keys):
    """Keep one doc per exact hash and per near-dup key, preferring the longest.

    seen_hashes / seen_keys are updated in place so later calls respect earlier ones.
    """
    kept = []
    for i in sorted(indices, key=lambda j: (-len(docs[j]), j)):
        h, k = _dedupe_keys(docs[i])
        if h in seen_hashes or k in seen_keys:
            continue
        seen_hashes.add(h)
        seen_keys.add(k)
        kept.append(i)
    kept.sort()
    return kept


class _TokenCounter:
    """tiktoken gpt2 counts, falling back to chars/4 when time is tight."""

    def __init__(self, deadline):
        self.deadline = deadline
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None

    def count(self, texts):
        if self.enc is not None and time.time() < self.deadline:
            try:
                return [len(x) for x in self.enc.encode_ordinary_batch(texts)]
            except Exception:
                pass
        return [max(1, len(t) // 4) for t in texts]

    def total(self, indices, docs, batch=256):
        tot = 0
        for s in range(0, len(indices), batch):
            tot += sum(self.count([docs[i] for i in indices[s:s + batch]]))
        return tot


# ------------------------------------------------------------------ select

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback():
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    if n == 0:
        return []

    # --- 2 (setup): windowed bigram scorer + leave-one-out threshold from the target
    target_bodies = [_split_title(t)[1] for t in target_sample if isinstance(t, str)]
    ws_score, ws_loo = _build_window_scorer(target_bodies, _tok_ws)
    ws_thresh = float(np.percentile(ws_loo, 5)) * 0.8 if ws_loo else 0.0

    # --- 3: junk filter over the whole pool (needed for both survivors and fill)
    non_junk = []
    for i, d in enumerate(docs):
        if isinstance(d, str) and not _is_junk(d):
            non_junk.append(i)

    # --- 1 + 2: format gate and whole-document (min-window) damage filter
    survivors = []
    rest = []
    for i in non_junk:
        d = docs[i]
        if _format_ok(d) and ws_score(_split_title(d)[1]) >= ws_thresh:
            survivors.append(i)
        else:
            rest.append(i)

    # --- 4: dedupe
    seen_hashes, seen_keys = set(), set()
    survivors = _dedupe(survivors, docs, seen_hashes, seen_keys)

    # --- 5: order and fill
    counter = _TokenCounter(deadline=t0 + _TIME_BUDGET * 0.6)
    total = counter.total(survivors, docs)

    if total >= 0.5 * T:
        rng.shuffle(survivors)
        return survivors

    # Not enough in-domain text: top up with clean out-of-domain docs.
    alpha_score, alpha_loo = _build_scorer(target_bodies, _tok_alpha)
    alpha_floor = float(np.percentile(alpha_loo, 5)) * 0.5 if alpha_loo else 0.0

    rest = _dedupe(rest, docs, seen_hashes, seen_keys)
    scored = []
    for i in rest:
        s = alpha_score(_split_title(docs[i])[1])
        if s >= alpha_floor:  # below the floor = shuffled words / not prose
            scored.append((s, i))
    scored.sort(key=lambda x: (-x[0], x[1]))

    counter.deadline = t0 + _TIME_BUDGET
    out = list(survivors)
    batch = 256
    pos = 0
    while total < T and pos < len(scored):
        chunk = [i for _, i in scored[pos:pos + batch]]
        pos += batch
        for i, c in zip(chunk, counter.count([docs[i] for i in chunk])):
            if total >= T:
                break
            out.append(i)
            total += c

    if not out:
        return _fallback()

    rng.shuffle(out)
    return out
