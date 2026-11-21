"""Curator: presence-only vocabulary filter plus per-batch document diversity.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Score each document on the *set* of distinct words in its first N_WORDS
     words (presence only, no frequency ratios):
       - miss fraction: share of its distinct words that never occur in the
         target sample. A small tolerance is allowed, because even genuine
         target documents contain some words unseen in a 200-document sample.
         The tolerance is estimated leave-one-out on the target sample itself.
         Anything beyond it is penalised heavily.
       - boost: share of its distinct words that are high-frequency target
         words (words occurring in many target documents).
  3. Fill the budget of T tokens greedily, one batch (about one training
     step's worth of tokens) at a time. Within the current batch a candidate
     is penalised when its content-word set (distinct words that are not
     high-frequency target words) significantly overlaps with the words of
     the documents already chosen for that batch. The greedy is lazy: the
     best candidate is popped from a heap, its penalty is recomputed, and it
     is selected only if it still beats the next candidate; otherwise it is
     pushed back. Penalties are cleared when a batch closes, so a deferred
     document competes again at its base score in the next batch.
  4. Shuffle batch order and the order inside each batch (seeded by
     CURATE_SEED), keeping the batches intact; the unchosen documents follow
     in score order as a backstop.
"""
import os
import re
import time
import heapq
import random
import hashlib
from array import array
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

N_WORDS = 400          # words per document used for scoring
CHAR_CAP = 6000        # characters per document scanned for those words
HF_MIN_DOC_FRAC = 0.05  # a word is "high-frequency" if in >= this share of target docs
HF_MIN_DOCS = 3         # ... and in at least this many target docs
TOL_QUANTILE = 0.75     # quantile of leave-one-out target miss rates used as tolerance
TOL_LO, TOL_HI = 0.02, 0.60
EXCESS_PENALTY = 6.0    # per unit of miss fraction beyond the tolerance
BASE_PENALTY = 0.5      # mild slope on miss fraction inside the tolerance
BOOST_WEIGHT = 1.0      # weight of the high-frequency-target-word share
BUDGET_MARGIN = 1.03   # select slightly more than T to absorb count mismatches
CHUNK = 2000           # documents per tokeniser batch

# ---- diversity ----
BATCH_TOKENS = max(1, T // 300)  # one training step's worth of tokens
DIV_THRESHOLD = 0.5    # overlap share above which the penalty starts
DIV_WEIGHT = 0.3       # penalty at full overlap
DIV_MIN_WORDS = 8      # minimum content words for the overlap to be meaningful
TIME_GUARD = 110.0     # seconds after which the diversity penalty is switched off

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _words(text):
    return _WORD_RE.findall(text[:CHAR_CAP].lower())[:N_WORDS]


def _make_counter():
    """Return a function mapping a list of strings to a list of token counts."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            try:
                return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
            except Exception:
                out = []
                for t in texts:
                    try:
                        out.append(len(enc.encode_ordinary(t)))
                    except Exception:
                        out.append(max(1, len(t) // 4))
                return out

        return count
    except Exception:
        return lambda texts: [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- target statistics (presence only: document frequencies) ----
    t_df = Counter()
    t_sets = []
    t_lens = []
    for s in target_sample:
        w = _words(s) if isinstance(s, str) else []
        t_lens.append(len(w))
        if w:
            ws = set(w)
            t_sets.append(ws)
            t_df.update(ws)

    if not t_df:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    n_t = len(t_sets)
    t_vocab = set(t_df)

    # high-frequency target words: present in many target documents
    hf_min = max(HF_MIN_DOCS, int(HF_MIN_DOC_FRAC * n_t + 0.999))
    hf_vocab = {w for w, c in t_df.items() if c >= hf_min}
    if not hf_vocab:
        # tiny target sample: fall back to the most widespread words
        hf_vocab = {w for w, _ in t_df.most_common(200)}

    # tolerance: how many unseen words does a genuine target doc have when
    # compared against the rest of the sample (leave-one-out)?
    loo = []
    for ws in t_sets:
        if len(ws) >= 5:
            miss = sum(1 for w in ws if t_df[w] <= 1)
            loo.append(miss / len(ws))
    if loo:
        loo.sort()
        tol = loo[min(len(loo) - 1, int(TOL_QUANTILE * len(loo)))]
    else:
        tol = 0.2
    tol = min(TOL_HI, max(TOL_LO, tol))

    t_lens.sort()
    # minimum words a document needs, adapted to how short target docs are
    min_words = max(3, min(20, (t_lens[len(t_lens) // 10] if t_lens else 20) // 2))

    # ---- pool pass: dedupe, filter, score, collect content-word ids ----
    seen = set()
    scored = []            # (score, doc index, slot)
    vocab = {}             # content word -> integer id
    flat = array("i")      # concatenated content-word ids
    offs = [0]             # slot -> start offset into flat
    for i, d in enumerate(docs):
        if not isinstance(d, str) or not d:
            continue
        key = hashlib.md5(
            _WS_RE.sub(" ", d).strip().lower().encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        w = _words(d)
        if len(w) < min_words:
            continue
        ws = set(w)
        # highly repetitive documents (boilerplate / junk)
        if len(w) >= 50 and len(ws) / len(w) < 0.2:
            continue
        nt = len(ws)
        present = ws & t_vocab
        miss_frac = 1.0 - len(present) / nt
        hf_frac = len(present & hf_vocab) / nt
        excess = miss_frac - tol
        s = BOOST_WEIGHT * hf_frac - BASE_PENALTY * miss_frac
        if excess > 0:
            s -= EXCESS_PENALTY * excess
        content = ws - hf_vocab
        flat.extend([vocab.setdefault(x, len(vocab)) for x in content])
        scored.append((s, i, len(offs) - 1))
        offs.append(len(flat))

    if not scored:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # deterministic ordering: score descending, index as tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))
    m = len(scored)
    ranked = [i for _, i, _ in scored]
    base = [s for s, _, _ in scored]
    slots = [k for _, _, k in scored]

    flat_np = np.frombuffer(flat, dtype=np.int32) if len(flat) else np.zeros(0, dtype=np.int32)
    in_batch = np.zeros(len(vocab) + 1, dtype=bool)

    # ---- lazy token counting by rank position ----
    count = _make_counter()
    lens = [-1] * m

    def length_of(p):
        if lens[p] < 0:
            a = (p // CHUNK) * CHUNK
            b = min(m, a + CHUNK)
            for q, L in zip(range(a, b), count([docs[ranked[q]] for q in range(a, b)])):
                lens[q] = L
        return lens[p]

    # ---- greedy, diversity-penalised fill of the token budget ----
    budget = int(T * BUDGET_MARGIN)
    cur = [-b for b in base]           # current valid heap key per position (None = chosen)
    heap = [(-base[p], p) for p in range(m)]  # already sorted => valid heap
    batches = []
    batch = []
    batch_tok = 0
    demoted = []
    total = 0
    use_div = DIV_WEIGHT > 0
    pops = 0
    span = 1.0 - DIV_THRESHOLD

    while heap and total < budget:
        negk, p = heapq.heappop(heap)
        if cur[p] is None or cur[p] != negk:
            continue
        pops += 1
        if use_div and (pops & 4095) == 0 and time.time() - t_start > TIME_GUARD:
            use_div = False

        k = slots[p]
        a, b = offs[k], offs[k + 1]
        ids = flat_np[a:b]
        if use_div and batch and (b - a) >= DIV_MIN_WORDS:
            overlap = int(np.count_nonzero(in_batch[ids])) / (b - a)
            if overlap > DIV_THRESHOLD:
                pen = DIV_WEIGHT * (overlap - DIV_THRESHOLD) / span
                newneg = pen - base[p]
                if heap and newneg > heap[0][0]:
                    # no longer the best candidate: defer
                    cur[p] = newneg
                    heapq.heappush(heap, (newneg, p))
                    demoted.append(p)
                    continue

        # select
        L = length_of(p)
        cur[p] = None
        batch.append(ranked[p])
        total += L
        batch_tok += L
        if b > a:
            in_batch[ids] = True

        if batch_tok >= BATCH_TOKENS:
            batches.append(batch)
            batch = []
            batch_tok = 0
            in_batch[:] = False
            # deferred documents compete again at their base score
            for q in demoted:
                c = cur[q]
                if c is not None and c != -base[q]:
                    cur[q] = -base[q]
                    heapq.heappush(heap, (-base[q], q))
            demoted = []

    if batch:
        batches.append(batch)

    if not batches:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # shuffle batch order and inside each batch, keeping batches intact
    rng.shuffle(batches)
    chosen = []
    for bt in batches:
        rng.shuffle(bt)
        chosen.extend(bt)

    # If the best documents fall short of T, the remaining ones (still in score
    # order) top the stream up before any cycling happens.
    rest = [ranked[p] for p in range(m) if cur[p] is not None]
    return chosen + rest
