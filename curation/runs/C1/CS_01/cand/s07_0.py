"""Curator: minimum-domain-coverage filter against the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. For each document take the *set* of distinct words in its first N_WORDS
     words and compute:
       - domain specificity: share of its distinct words that occur anywhere
         in the target sample.
       - boost: share of its distinct words that are high-frequency target
         words (words occurring in many target documents).
     Only documents whose domain specificity reaches a hard threshold
     (MIN_COVERAGE, 0.6) are eligible. The threshold is lowered only if a
     leave-one-out check on the target sample shows that genuine target
     documents themselves would mostly fail it.
  3. Rank eligible documents by specificity + boost and walk from the top,
     counting tokens with the tiktoken "gpt2" encoder, until the budget of
     T tokens is filled.
  4. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
     mixed stream. Documents below the threshold are never included; if the
     eligible documents hold fewer than T tokens the trainer cycles them.
"""
import os
import re
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
MIN_COVERAGE = 0.6      # minimum domain specificity for a document to be eligible
COVERAGE_FLOOR = 0.3    # the adaptive safeguard never lowers the threshold below this
LOO_QUANTILE = 0.25     # target docs at this leave-one-out quantile must still pass
BOOST_WEIGHT = 1.0      # weight of the high-frequency-target-word share
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

    # Safeguard for the hard threshold: the coverage a genuine target doc gets
    # against the rest of the sample (leave-one-out). If most target docs
    # could not reach MIN_COVERAGE themselves, lower the threshold.
    loo_cov = []
    for ws in t_sets:
        if len(ws) >= 5:
            hit = sum(1 for w in ws if t_df[w] >= 2)
            loo_cov.append(hit / len(ws))
    threshold = MIN_COVERAGE
    if loo_cov:
        loo_cov.sort()
        q = loo_cov[min(len(loo_cov) - 1, int(LOO_QUANTILE * len(loo_cov)))]
        threshold = min(MIN_COVERAGE, max(COVERAGE_FLOOR, q))

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
        present = ws & t_vocab
        specificity = len(present) / nt
        # Minimum Domain Coverage: hard cut, no continuous penalty
        if specificity < threshold:
            continue
        hf_frac = len(present & hf_vocab) / nt
        s = specificity + BOOST_WEIGHT * hf_frac
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

    # Only documents above the coverage threshold are returned; if they hold
    # fewer than T tokens the trainer cycles this list.
    rng.shuffle(chosen)
    return chosen
