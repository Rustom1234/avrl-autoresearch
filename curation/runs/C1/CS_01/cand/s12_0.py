"""Curator: target word density scoring against the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Score each document on the *set* of distinct words in its first N_WORDS
     words using "Target Word Density":
       - every word that occurs in the target sample has an importance weight
         1 / sqrt(target_freq + 1), where target_freq is the number of target
         documents containing the word;
       - words that never occur in the target sample contribute nothing (no
         explicit penalty for "missing" words);
       - score = (sum of weights of the document's distinct words that are
         present in the target sample) / (number of distinct words in the
         document).
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

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

N_WORDS = 400          # words per document used for scoring
CHAR_CAP = 6000        # characters per document scanned for those words
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

    # ---- target statistics: document frequency of each word ----
    t_df = Counter()
    t_lens = []
    for s in target_sample:
        w = _words(s) if isinstance(s, str) else []
        t_lens.append(len(w))
        if w:
            t_df.update(set(w))

    if not t_df:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # importance weight of each target-present word
    weight = {w: 1.0 / ((c + 1.0) ** 0.5) for w, c in t_df.items()}
    wget = weight.get

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
        # Target Word Density: weighted mass of target-present words per
        # distinct word in the document.
        s = sum(wget(x, 0.0) for x in ws) / nt
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
