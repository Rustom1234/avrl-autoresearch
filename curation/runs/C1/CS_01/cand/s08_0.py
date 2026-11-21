"""Curator: presence-only vocabulary filter with document length normalisation.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Score each document on the *set* of distinct words in its first N_WORDS
     words (presence only, no frequency ratios):
       - miss fraction: share of its distinct words that never occur in the
         target sample. A small tolerance is allowed, estimated leave-one-out
         on the target sample itself. Anything beyond it is penalised heavily.
       - boost: share of its distinct words that are high-frequency target
         words (words occurring in many target documents).
  3. Document length normalisation (token lengths from the tiktoken "gpt2"
     encoder):
       - the miss-fraction penalties are multiplied by a weight that grows
         with the document's token length relative to the target mean
         (a long off-vocabulary document wastes more of the budget);
       - documents whose token length lies within one standard deviation of
         the target sample's mean token length get a priority bonus, which
         decays (and turns into a penalty) further out.
  4. Walk the documents from highest score down until the budget of T tokens
     is filled.
  5. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
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
HF_MIN_DOC_FRAC = 0.05  # a word is "high-frequency" if in >= this share of target docs
HF_MIN_DOCS = 3         # ... and in at least this many target docs
TOL_QUANTILE = 0.75     # quantile of leave-one-out target miss rates used as tolerance
TOL_LO, TOL_HI = 0.02, 0.60
EXCESS_PENALTY = 6.0    # per unit of miss fraction beyond the tolerance
BASE_PENALTY = 0.5      # mild slope on miss fraction inside the tolerance
BOOST_WEIGHT = 1.0      # weight of the high-frequency-target-word share
LEN_W_LO, LEN_W_HI = 0.5, 2.0  # clamp on the length weight of the miss penalty
LEN_BONUS = 0.15        # bonus for token length within 1 sd of the target mean
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

    # ---- target statistics (presence only: document frequencies) ----
    t_df = Counter()
    t_sets = []
    t_lens = []
    t_texts = []
    for s in target_sample:
        if isinstance(s, str) and s:
            t_texts.append(s)
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

    # ---- target token-length statistics ----
    count = _make_counter()
    t_tok = [L for L in count(t_texts) if L > 0] if t_texts else []
    if t_tok:
        mu = sum(t_tok) / len(t_tok)
        var = sum((L - mu) ** 2 for L in t_tok) / len(t_tok)
        sd = math.sqrt(var)
    else:
        mu, sd = 0.0, 0.0
    use_len = mu > 0
    if use_len:
        sd = max(sd, 0.1 * mu, 1.0)

    # ---- pool pass: dedupe, filter, vocabulary features ----
    seen = set()
    feats = []  # (index, miss_frac, hf_frac)
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
        feats.append((i, miss_frac, hf_frac))

    if not feats:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- token lengths of all candidate documents ----
    tok_len = {}
    for pos in range(0, len(feats), CHUNK):
        batch = [f[0] for f in feats[pos:pos + CHUNK]]
        lens = count([docs[i] for i in batch])
        for i, L in zip(batch, lens):
            tok_len[i] = max(1, L)

    # ---- scoring with document length normalisation ----
    scored = []
    for i, miss_frac, hf_frac in feats:
        L = tok_len[i]
        if use_len:
            # miss penalty weighted by token length (relative to target mean)
            len_w = min(LEN_W_HI, max(LEN_W_LO, math.sqrt(L / mu)))
            # priority for lengths within one sd of the target mean:
            # +1 for z <= 1, 0 at z = 2, -1 for z >= 3
            z = abs(L - mu) / sd
            len_term = max(-1.0, min(1.0, 2.0 - z))
        else:
            len_w = 1.0
            len_term = 0.0
        s = BOOST_WEIGHT * hf_frac - len_w * BASE_PENALTY * miss_frac
        excess = miss_frac - tol
        if excess > 0:
            s -= len_w * EXCESS_PENALTY * excess
        s += LEN_BONUS * len_term
        scored.append((s, i))

    # deterministic ordering: score descending, index as tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

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
