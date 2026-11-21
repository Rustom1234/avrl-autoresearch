"""Curator: dynamic word importance (local IDF) against the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Score each document on the *set* of distinct words in its first N_WORDS
     words:
       - miss fraction: share of its distinct words that never occur in the
         target sample. A small tolerance is allowed, because even genuine
         target documents contain some words unseen in a 200-document sample.
         The tolerance is estimated leave-one-out on the target sample itself.
         Anything beyond it is penalised heavily.
       - dynamic word importance: every distinct word shared with the target
         sample contributes a local-IDF weight
             1 / (sqrt(count_in_target_sample) + 1)
         so rarer (more topical) target words count more than ubiquitous
         function words. The sum is normalised by the document's number of
         distinct words.
  3. Walk the documents from highest score down, counting tokens with the
     tiktoken "gpt2" encoder, until the budget of T tokens is filled.
  4. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
     mixed stream; the unchosen documents follow in score order as a backstop.
"""
import os
import re
import math
import random
import hashlib
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

N_WORDS = 400          # words per document used for scoring
CHAR_CAP = 6000        # characters per document scanned for those words
TOL_QUANTILE = 0.75     # quantile of leave-one-out target miss rates used as tolerance
TOL_LO, TOL_HI = 0.02, 0.60
EXCESS_PENALTY = 6.0    # per unit of miss fraction beyond the tolerance
BASE_PENALTY = 0.5      # mild slope on miss fraction inside the tolerance
IMPORTANCE_WEIGHT = 1.0  # weight of the normalised local-IDF importance sum
BUDGET_MARGIN = 1.03   # select slightly more than T to absorb count mismatches
CHUNK = 2000           # documents per tokeniser batch

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
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- target statistics ----
    t_df = Counter()     # document frequency (for leave-one-out tolerance)
    t_count = Counter()  # total occurrences in the target sample (for local IDF)
    t_sets = []
    t_lens = []
    for s in target_sample:
        w = _words(s) if isinstance(s, str) else []
        t_lens.append(len(w))
        if w:
            ws = set(w)
            t_sets.append(ws)
            t_df.update(ws)
            t_count.update(w)

    if not t_df:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # dynamic word importance: local IDF weight per target word
    importance = {w: 1.0 / (math.sqrt(c) + 1.0) for w, c in t_count.items()}
    imp_get = importance.get

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

    # ---- pool pass: dedupe, filter, score ----
    seen = set()
    scored = []
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
        n_present = 0
        imp_sum = 0.0
        for x in ws:
            v = imp_get(x)
            if v is not None:
                n_present += 1
                imp_sum += v
        miss_frac = 1.0 - n_present / nt
        imp_score = imp_sum / nt
        excess = miss_frac - tol
        s = IMPORTANCE_WEIGHT * imp_score - BASE_PENALTY * miss_frac
        if excess > 0:
            s -= EXCESS_PENALTY * excess
        scored.append((s, i))

    if not scored:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # deterministic ordering: score descending, index as tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

    # ---- fill the token budget from the top of the ranking ----
    count = _make_counter()
    budget = int(T * BUDGET_MARGIN)
    chosen = []
    total = 0
    pos = 0
    while pos < len(ranked) and total < budget:
        batch = ranked[pos:pos + CHUNK]
        lens = count([docs[i] for i in batch])
        for i, L in zip(batch, lens):
            if total >= budget:
                break
            chosen.append(i)
            total += L
            pos += 1

    if not chosen:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    rest = ranked[len(chosen):]
    rng.shuffle(chosen)
    # If the best documents fall short of T, the remaining ones (still in score
    # order) top the stream up before any cycling happens.
    return chosen + rest
