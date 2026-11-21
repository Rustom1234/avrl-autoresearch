"""Curator: junk guards + dedupe + word-order filter + weighted vocab/bigram score.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  1. dedupe      : key = normalised lowercased first 500 chars + length bucket
  2. junk guards : too short / too few alphabetic chars / repetitive
  3. word-order  : hit rate of common-word bigrams against the target's bigram set
  4. score        : weighted sum (Bigram_Hit_Rate * 0.7) + (Vocab_Overlap * 0.3)
  5. selection    : p >= 0.5, capped at T tokens, topped up to 0.6 T if short; shuffled
"""
import os
import random
import re
import time
from collections import Counter
from itertools import repeat

import numpy as np

from common import T

# ---- tunables -------------------------------------------------------------
DEDUPE_PREFIX = 500        # chars used for the dedupe key
MIN_CHARS = 200            # junk guard: minimum document length
MIN_ALPHA_FRAC = 0.60      # junk guard: alphabetic-or-space fraction
MIN_UNIQ_RATIO = 0.20      # junk guard: unique words / words (first WORDS_CAP words)
TOP_WORDS = 3000           # vocabulary for the bigram filter
VOCAB_LIMIT = 5000         # vocabulary for the overlap score
WORDS_CAP = 2000           # words per doc inspected by the bigram filter
MIN_BIGRAMS = 10           # docs with fewer in-vocab bigrams get hit rate 0
CAL_PERCENTILE = 2.0       # percentile of held-out target hit rates
CAL_SLACK = 0.9            # multiplier on that percentile
SCORE_CHAR_BUDGET = 30_000_000  # total chars vectorised when scoring (time guard)
P_KEEP = 0.5               # probability threshold
TOPUP_FRAC = 0.6           # minimum fraction of T to hand to the trainer
TOK_BATCH = 2000           # docs tokenised per batch

_WS = re.compile(r"\s+")
_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")

def _log(msg):
    print("[curate] " + msg, flush=True)

def _words(text, cap=WORDS_CAP):
    if not isinstance(text, str):
        return []
    ws = text[: cap * 12].lower().split()
    return ws[:cap] if cap else ws

def _all_words(text):
    if not isinstance(text, str):
        return []
    return text.lower().split()

def _bigram_codes(ws, vocab, V):
    """Flat codes a*V+b for adjacent word pairs where both words are in vocab."""
    if len(ws) < 2:
        return np.empty(0, dtype=np.int64)
    ids = np.fromiter(map(lambda x: vocab.get(x, -1), ws), dtype=np.int64, count=len(ws))
    a = ids[:-1]
    b = ids[1:]
    m = (a >= 0) & (b >= 0)
    return a[m] * V + b[m]

def _build_table(texts, vocab, V):
    table = np.zeros(V * V, dtype=bool)
    for t in texts:
        if not isinstance(t, str): continue
        codes = _bigram_codes(_all_words(t), vocab, V)
        if codes.size:
            table[codes] = True
    return table

def _hit_rate(ws, vocab, V, table):
    codes = _bigram_codes(ws, vocab, V)
    if codes.size < MIN_BIGRAMS:
        return 0.0
    return float(table[codes].mean())

def _chunks(text, size=2000):
    out = []
    for s in range(0, len(text), size):
        c = text[s : s + size]
        if s == 0 or len(c) >= size // 4:
            out.append(c)
    return out

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)
    _log(f"pool docs: {n_docs}, target docs: {len(target_sample)}, T={T}")

    # Build Reference Vocabulary
    wc = Counter()
    for t in target_sample:
        wc.update(_all_words(t))
    
    # Vocab for Bigram Hit Rate (TOP_WORDS)
    bigram_vocab = {w: i for i, (w, _) in enumerate(wc.most_common(TOP_WORDS))}
    V = max(len(bigram_vocab), 1)
    
    # Vocab for Overlap Score (VOCAB_LIMIT)
    overlap_vocab = set(w for w, _ in wc.most_common(VOCAB_LIMIT))

    # Calibration
    cutoff = 0.0
    half = len(target_sample) // 2
    if half >= 2:
        cal_table = _build_table(target_sample[:half], bigram_vocab, V)
        cal_scores = [
            _hit_rate(_words(t), bigram_vocab, V, cal_table) for t in target_sample[half:]
        ]
        cal = np.asarray(cal_scores, dtype=np.float64)
        cutoff = float(np.percentile(cal, CAL_PERCENTILE)) * CAL_SLACK
        _log(f"bigram calibration -> cutoff={cutoff:.3f}")
    
    table = _build_table(target_sample, bigram_vocab, V)

    # Filters
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if not isinstance(d, str): continue
        L = len(d)
        head = _WS.sub(" ", d[: DEDUPE_PREFIX * 2].lower()).strip()[:DEDUPE_PREFIX]
        key = (head, L.bit_length())
        if key in seen: continue
        seen.add(key)

        if L < MIN_CHARS: continue
        sample = d[:5000]
        non_alpha = len(_ALPHA_SPACE.sub("", sample))
        if 1.0 - non_alpha / len(sample) < MIN_ALPHA_FRAC: continue
        
        ws = _words(d)
        if len(ws) < 20 or (len(ws) >= 50 and len(set(ws)) / len(ws) < MIN_UNIQ_RATIO):
            continue
        
        hr = _hit_rate(ws, bigram_vocab, V, table)
        if hr < cutoff: continue
        
        survivors.append(i)
    del seen

    _log(f"survivors after filters: {len(survivors)} ({time.time() - t0:.1f}s)")

    if not survivors:
        idx = list(range(n_docs))
        rng.shuffle(idx)
        return idx

    # Scoring
    scores = []
    for i in survivors:
        d = docs[i]
        ws = _words(d)
        doc_vocab = set(_all_words(d[:WORDS_CAP * 12]))
        inter = len(overlap_vocab.intersection(doc_vocab))
        denom = max(len(overlap_vocab), len(doc_vocab))
        overlap_score = inter / denom if denom > 0 else 0
        
        hr = _hit_rate(ws, bigram_vocab, V, table)
        final_score = (hr * 0.7) + (overlap_score * 0.3)
        scores.append(final_score)

    _log(f"score distribution: p5={np.percentile(scores, 5):.3f}, p50={np.percentile(scores, 50):.3f}, p95={np.percentile(scores, 95):.3f}")

    # Selection
    sorted_indices = np.argsort(-np.array(scores), kind="stable")
    ranked = [survivors[j] for j in sorted_indices]
    ranked_scores = np.array(scores)[sorted_indices]
    
    n_pos = int((ranked_scores >= P_KEEP).sum())
    
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count_tokens(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
    except:
        def count_tokens(texts):
            return [max(1, len(x) // 4) for x in texts]

    topup_target = int(TOPUP_FRAC * T)
    selected = []
    tot = 0
    for s in range(0, len(ranked), TOK_BATCH):
        batch_idx = ranked[s : s + TOK_BATCH]
        batch_texts = [docs[i] for i in batch_idx]
        counts = count_tokens(batch_texts)
        for k, (idx, n) in enumerate(zip(batch_idx, counts)):
            pos_idx = s + k
            if pos_idx < n_pos:
                if tot >= T: break
                selected.append(idx)
                tot += n
            else:
                if tot >= topup_target: break
                selected.append(idx)
                tot += n
        if tot >= T or (tot >= topup_target and len(selected) >= n_pos):
            break

    _log(f"final: {len(selected)} docs, {tot} tokens ({tot / T:.2f} T)")
    
    if not selected:
        selected = list(survivors)
    rng.shuffle(selected)
    return selected
