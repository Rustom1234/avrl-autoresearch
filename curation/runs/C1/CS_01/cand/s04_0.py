"""Curator: presence-only vocabulary filter against the target sample,
with a document-length weighting factor.

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
  3. Document-length weighting: token counts (tiktoken "gpt2") of the target
     sample define a preferred band around the target's average length.
     Documents inside the band get the full length factor (1.0); outside it,
     the factor decays smoothly (Gaussian in log-length), so extremely short
     or excessively long documents are pushed down the ranking.
  4. Walk the documents from highest score down, counting tokens, until the
     budget of T tokens is filled.
  5. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
     mixed stream; the unchosen documents follow in score order as a backstop.
"""
import os
import re
import math
import time
import random
import hashlib
from collections import Counter

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

# ---- document-length weighting ----
LEN_WEIGHT = 0.5        # max score penalty for a badly mis-sized document
LEN_BAND_FACTOR = 3.0   # band is at least [mean / factor, mean * factor]
LEN_Q_LO, LEN_Q_HI = 0.10, 0.90  # ... widened to these target length quantiles
LEN_SIGMA = 1.0         # decay width outside the band, in log-length units
TOKENISE_DEADLINE = 100.0  # seconds from start after which lengths are estimated

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


def _length_band(target_tok_lens):
    """Preferred token-length band (lo, hi) built around the target average."""
    lens = sorted(L for L in target_tok_lens if L > 0)
    if not lens:
        return None
    mean = sum(lens) / len(lens)
    q_lo = lens[min(len(lens) - 1, int(LEN_Q_LO * len(lens)))]
    q_hi = lens[min(len(lens) - 1, int(LEN_Q_HI * len(lens)))]
    lo = max(1.0, min(q_lo, mean / LEN_BAND_FACTOR))
    hi = max(q_hi, mean * LEN_BAND_FACTOR, lo + 1.0)
    return lo, hi


def _length_factor(L, band):
    """1.0 inside the band, smooth decay towards 0 outside it."""
    if band is None:
        return 1.0
    lo, hi = band
    L = max(1, L)
    if L < lo:
        d = math.log(lo / L)
    elif L > hi:
        d = math.log(L / hi)
    else:
        return 1.0
    return math.exp(-(d * d) / (2.0 * LEN_SIGMA * LEN_SIGMA))


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

    # ---- target token lengths -> preferred length band ----
    count = _make_counter()
    t_texts = [s for s in target_sample if isinstance(s, str) and s]
    try:
        t_tok_lens = count(t_texts) if t_texts else []
    except Exception:
        t_tok_lens = [max(1, len(s) // 4) for s in t_texts]
    band = _length_band(t_tok_lens)

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
        present = ws & t_vocab
        miss_frac = 1.0 - len(present) / nt
        hf_frac = len(present & hf_vocab) / nt
        excess = miss_frac - tol
        s = BOOST_WEIGHT * hf_frac - BASE_PENALTY * miss_frac
        if excess > 0:
            s -= EXCESS_PENALTY * excess
        scored.append((s, i))

    if not scored:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- token counts for surviving documents (time-guarded) ----
    # Best vocabulary scores are tokenised first so that, if the deadline
    # hits, exact counts exist for the documents most likely to be chosen.
    scored.sort(key=lambda x: (-x[0], x[1]))
    tok_len = {}
    chars_done = 0
    toks_done = 0
    pos = 0
    while pos < len(scored):
        if time.time() - t_start > TOKENISE_DEADLINE:
            break
        batch = [i for _, i in scored[pos:pos + CHUNK]]
        texts = [docs[i] for i in batch]
        lens = count(texts)
        for i, t, L in zip(batch, texts, lens):
            tok_len[i] = L
            chars_done += len(t)
            toks_done += L
        pos += len(batch)
    if pos < len(scored):
        # estimate the remainder from the observed characters-per-token ratio
        ratio = (chars_done / toks_done) if toks_done > 0 else 4.0
        ratio = max(1.0, ratio)
        for _, i in scored[pos:]:
            tok_len[i] = max(1, int(len(docs[i]) / ratio))

    # ---- apply the document-length weighting factor ----
    rescored = []
    for s, i in scored:
        f = _length_factor(tok_len[i], band)
        rescored.append((s - LEN_WEIGHT * (1.0 - f), i))

    # deterministic ordering: score descending, index as tie-break
    rescored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in rescored]

    # ---- fill the token budget from the top of the ranking ----
    budget = int(T * BUDGET_MARGIN)
    chosen = []
    total = 0
    for i in ranked:
        if total >= budget:
            break
        chosen.append(i)
        total += tok_len[i]

    if not chosen:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    rest = ranked[len(chosen):]
    rng.shuffle(chosen)
    # If the best documents fall short of T, the remaining ones (still in score
    # order) top the stream up before any cycling happens.
    return chosen + rest
