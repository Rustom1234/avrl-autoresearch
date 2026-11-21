"""Curator: target-document Jaccard similarity with a high-frequency booster.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Score each document on the *set* of distinct words in its first N_WORDS
     words:
       - primary: average Jaccard similarity between the document's word set
         and the word set of every target-sample document. It is normalised
         by the typical leave-one-out average Jaccard of the target documents
         among themselves, so a "target-like" document scores about 1.
       - boost: share of its distinct words that are high-frequency target
         words (words occurring in many target documents).
       - guard: documents whose share of words never seen in the target
         sample exceeds a leave-one-out tolerance are penalised (junk /
         foreign-domain backstop).
  3. Walk the documents from highest score down, counting tokens with the
     tiktoken "gpt2" encoder, until the budget of T tokens is filled.
  4. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
     mixed stream; the unchosen documents follow in score order as a backstop.
"""
import os
import re
import random
import hashlib
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
JACCARD_WEIGHT = 2.0    # weight of the normalised average Jaccard (primary)
BOOST_WEIGHT = 0.5      # weight of the high-frequency-target-word share
EXCESS_PENALTY = 6.0    # per unit of miss fraction beyond the tolerance
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
    word_id = {w: k for k, w in enumerate(t_df)}
    n_vocab = len(word_id)

    # word x target-document incidence matrix
    M = np.zeros((n_vocab, n_t), dtype=np.float32)
    for j, ws in enumerate(t_sets):
        M[[word_id[w] for w in ws], j] = 1.0
    t_sizes = np.array([len(ws) for ws in t_sets], dtype=np.float32)

    # high-frequency target words: present in many target documents
    hf_min = max(HF_MIN_DOCS, int(HF_MIN_DOC_FRAC * n_t + 0.999))
    hf_vocab = {w for w, c in t_df.items() if c >= hf_min}
    if not hf_vocab:
        hf_vocab = {w for w, _ in t_df.most_common(200)}
    hf_flag = np.zeros(n_vocab, dtype=np.float32)
    for w in hf_vocab:
        hf_flag[word_id[w]] = 1.0

    # leave-one-out references on the target sample itself:
    #  - miss tolerance (share of words unseen in the other target docs)
    #  - typical average Jaccard of a target doc to the other target docs
    loo_miss = []
    loo_jac = []
    for j, ws in enumerate(t_sets):
        if len(ws) < 5:
            continue
        loo_miss.append(sum(1 for w in ws if t_df[w] <= 1) / len(ws))
        if n_t > 1:
            ids = [word_id[w] for w in ws]
            inter = M[ids].sum(axis=0)
            jac = inter / (len(ws) + t_sizes - inter)
            loo_jac.append(float((jac.sum() - jac[j]) / (n_t - 1)))
    if loo_miss:
        loo_miss.sort()
        tol = loo_miss[min(len(loo_miss) - 1, int(TOL_QUANTILE * len(loo_miss)))]
    else:
        tol = 0.2
    tol = min(TOL_HI, max(TOL_LO, tol))
    jac_ref = float(np.median(loo_jac)) if loo_jac else 0.0
    if jac_ref <= 1e-6:
        jac_ref = 0.05

    t_lens.sort()
    # minimum words a document needs, adapted to how short target docs are
    min_words = max(3, min(20, (t_lens[len(t_lens) // 10] if t_lens else 20) // 2))

    # ---- pool pass: dedupe, filter, score ----
    seen = set()
    scored = []
    get_id = word_id.get
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
        ids = [k for k in map(get_id, ws) if k is not None]
        if ids:
            inter = M[ids].sum(axis=0)
            avg_jac = float((inter / (nt + t_sizes - inter)).mean())
            hf_frac = float(hf_flag[ids].sum()) / nt
        else:
            avg_jac = 0.0
            hf_frac = 0.0
        miss_frac = 1.0 - len(ids) / nt
        s = JACCARD_WEIGHT * (avg_jac / jac_ref) + BOOST_WEIGHT * hf_frac
        excess = miss_frac - tol
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
