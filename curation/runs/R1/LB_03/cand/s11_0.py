"""Curator: junk guards + dedupe + word-order filter + structural proximity.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  1. dedupe      : key = normalised lowercased first 500 chars + length bucket
  2. junk guards : too short / too few alphabetic chars / repetitive
  3. word-order  : hit rate of common-word bigrams against the target's bigram set
  4. structural  : score based on target markers and local vocabulary density
  5. selection   : weighted score (0.5 bigram + 0.5 structural)
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
TOP_WORDS = 5000           # vocabulary for the bigram filter
WORDS_CAP = 2000           # words per doc inspected by the bigram filter
MIN_BIGRAMS = 10           # docs with fewer in-vocab bigrams get hit rate 0
CAL_PERCENTILE = 2.0       # percentile of held-out target hit rates
CAL_SLACK = 0.9            # multiplier on that percentile
TOK_BATCH = 2000           # docs tokenised per batch
TOPUP_FRAC = 0.6           # minimum fraction of T to hand to the trainer

_WS = re.compile(r"\s+")
_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")
_MARKERS = [r"@-@", r"=\s*Title\s*=", r"\(\s*book\s*\)"]

def _log(msg):
    print("[curate] " + msg, flush=True)

def _words(text, cap=WORDS_CAP):
    ws = text[: cap * 12].lower().split()
    return ws[:cap] if cap else ws

def _all_words(text):
    return text.lower().split()

def _bigram_codes(ws, vocab, V):
    if len(ws) < 2:
        return np.empty(0, dtype=np.int64)
    ids = np.fromiter(map(vocab.get, ws, repeat(-1)), dtype=np.int64, count=len(ws))
    a = ids[:-1]
    b = ids[1:]
    m = (a >= 0) & (b >= 0)
    return a[m] * V + b[m]

def _build_table(texts, vocab, V):
    table = np.zeros(V * V, dtype=bool)
    for t in texts:
        codes = _bigram_codes(_all_words(t), vocab, V)
        if codes.size:
            table[codes] = True
    return table

def _hit_rate(ws, vocab, V, table):
    codes = _bigram_codes(ws, vocab, V)
    if codes.size < MIN_BIGRAMS:
        return 0.0
    return float(table[codes].mean())

def _structural_score(text, vocab):
    score = 0.0
    # Find marker positions
    marker_indices = []
    for pattern in _MARKERS:
        for m in re.finditer(pattern, text):
            marker_indices.append(m.start())
    
    if not marker_indices:
        return 0.0
    
    # For each marker, check 10 words before/after
    # Estimation: 1 word ~ 6 chars
    win_size = 60 
    for idx in marker_indices:
        start = max(0, idx - win_size)
        end = min(len(text), idx + win_size)
        window_text = text[start:end].lower().split()
        
        # Count words in window that are in the top 5000 vocab
        hits = sum(1 for w in window_text if w in vocab)
        score += (hits / (len(window_text) + 1))
        
    return score / len(marker_indices)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)
    _log(f"pool docs: {n_docs}, target docs: {len(target_sample)}, T={T}")

    # 1. Build Vocab
    wc = Counter()
    for t in target_sample:
        wc.update(_all_words(t))
    vocab = {w: i for i, (w, _) in enumerate(wc.most_common(TOP_WORDS))}
    V = max(len(vocab), 1)

    # 2. Calibration
    cutoff = 0.0
    half = len(target_sample) // 2
    if half >= 2:
        cal_table = _build_table(target_sample[:half], vocab, V)
        cal_scores = [_hit_rate(_words(t), vocab, V, cal_table) for t in target_sample[half:]]
        cutoff = float(np.percentile(np.array(cal_scores), CAL_PERCENTILE)) * CAL_SLACK
        _log(f"bigram calibration cutoff: {cutoff:.3f}")
    table = _build_table(target_sample, vocab, V)

    # 3. Filter
    seen = set()
    survivors = []
    order_scores = []
    for i, d in enumerate(docs):
        if not isinstance(d, str): continue
        L = len(d)
        head = _WS.sub(" ", d[:DEDUPE_PREFIX*2].lower()).strip()[:DEDUPE_PREFIX]
        key = (head, L.bit_length())
        if key in seen: continue
        seen.add(key)
        if L < MIN_CHARS: continue
        sample = d[:5000]
        if 1.0 - len(_ALPHA_SPACE.sub("", sample)) / len(sample) < MIN_ALPHA_FRAC: continue
        ws = _words(d)
        if len(ws) < 20 or (len(ws) >= 50 and len(set(ws)) / len(ws) < MIN_UNIQ_RATIO): continue
        
        hr = _hit_rate(ws, vocab, V, table)
        order_scores.append(hr)
        if hr < cutoff: continue
        survivors.append(i)

    if not survivors:
        idx = list(range(n_docs))
        rng.shuffle(idx)
        return idx

    # 4. Scoring
    final_scores = []
    for i in survivors:
        d = docs[i]
        hr = _hit_rate(_words(d), vocab, V, table)
        ss = _structural_score(d, vocab)
        # Weighted sum
        final_scores.append(0.5 * hr + 0.5 * ss)
    
    final_scores = np.array(final_scores)
    order = np.argsort(-final_scores, kind="stable")
    ranked = [survivors[j] for j in order]
    ranked_p = final_scores[order]

    # 5. Selection
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count_tokens(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=1)]
    except:
        def count_tokens(texts):
            return [max(1, len(x) // 4) for x in texts]

    topup_target = int(TOPUP_FRAC * T)
    selected = []
    tot = 0
    n_pos = int((ranked_p >= 0.5).sum()) # Heuristic threshold for "pos" if needed
    
    # Simplified selection: take top by score until budget
    for i, doc_idx in enumerate(ranked):
        counts = count_tokens([docs[doc_idx]])
        n = counts[0]
        if tot >= T: break
        selected.append(doc_idx)
        tot += n
        
    _log(f"Selected {len(selected)} docs, {tot} tokens ({tot/T:.2f} T)")
    
    if not selected:
        selected = list(survivors)
    
    rng.shuffle(selected)
    return selected
