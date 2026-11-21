"""Curator: junk guards + dedupe + word-order filter + KL-Similarity.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  1. dedupe      : key = normalised lowercased first 500 chars + length bucket
  2. junk guards : too short / too few alphabetic chars / repetitive
  3. word order  : hit rate of common-word bigrams against the target's bigram set
  4. KL Similarity: log-likelihood of 1-gram distribution relative to target
     distribution (top 5000 words) + bigram hit rate weighted 0.6/0.4.
  5. selection   : p >= 0.5, capped at T tokens, topped up to 0.6 T if short; shuffled
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
TOP_WORDS_VAL = 3000       # vocabulary for the bigram filter
TOP_WORDS_KL = 5000        # vocabulary for the KL score
WORDS_CAP = 2000           # words per doc inspected by the bigram filter
MIN_BIGRAMS = 10           # docs with fewer in-vocab bigrams get hit rate 0
CAL_PERCENTILE = 2.0       # percentile of held-out target hit rates
CAL_SLACK = 0.9            # multiplier on that percentile
CHUNK_CHARS = 2000         # positive chunk size / negative truncation
MAX_FEATURES = 200_000
SCORE_CHAR_BUDGET = 30_000_000  # total chars vectorised when scoring (time guard)
P_KEEP = 0.5               # classifier probability threshold
TOPUP_FRAC = 0.6           # minimum fraction of T to hand to the trainer
TOK_BATCH = 2000           # docs tokenised per batch

_WS = re.compile(r"\s+")
_ALPHA_SPACE = re.compile(r"(?:[^\W\d_]|\s)+")


def _log(msg):
    print("[curate] " + msg, flush=True)


def _words(text, cap=WORDS_CAP):
    ws = text[: cap * 12].lower().split()
    return ws[:cap] if cap else ws


def _all_words(text):
    return text.lower().split()


def _bigram_codes(ws, vocab, V):
    """Flat codes a*V+b for adjacent word pairs where both words are in vocab."""
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


def _chunks(text, size=CHUNK_CHARS):
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

    # ------------------------------------------------------------------
    # 1) Bigram reference & KL Vocab
    # ------------------------------------------------------------------
    wc_bigram = Counter()
    wc_kl = Counter()
    for t in target_sample:
        ws = _all_words(t)
        wc_bigram.update(ws)
        wc_kl.update(ws)
    
    # Bigram vocab
    vocab_b = {w: i for i, (w, _) in enumerate(wc_bigram.most_common(TOP_WORDS_VAL))}
    V = max(len(vocab_b), 1)
    
    # KL vocab (top 5000)
    vocab_kl = {w: i for i, (w, _) in enumerate(wc_kl.most_common(TOP_WORDS_KL))}
    V_kl = max(len(vocab_kl), 1)
    
    # KL Probabilities (Multinomial approximation)
    counts_kl = np.array([wc_kl[w] for w in sorted(vocab_kl.keys())], dtype=np.float64)
    # Add tiny epsilon to avoid log(0)
    probs_kl = (counts_kl + 1e-9) / (counts_kl.sum() + 1e-9 * len(vocab_kl))

    cutoff = 0.0
    half = len(target_sample) // 2
    if half >= 2:
        cal_table = _build_table(target_sample[:half], vocab_b, V)
        cal_scores = [
            _hit_rate(_words(t), vocab_b, V, cal_table) for t in target_sample[half:]
        ]
        cal = np.asarray(cal_scores, dtype=np.float64)
        cutoff = float(np.percentile(cal, CAL_PERCENTILE)) * CAL_SLACK
        _log(f"bigram calibration: median={np.median(cal):.3f} -> cutoff={cutoff:.3f}")
    
    table = _build_table(target_sample, vocab_b, V)

    # ------------------------------------------------------------------
    # 2) Filters
    # ------------------------------------------------------------------
    seen = set()
    survivors = []
    order_scores = []
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
        
        hr = _hit_rate(ws, vocab_b, V, table)
        if hr < cutoff: continue
        
        order_scores.append(hr)
        survivors.append(i)
    
    del seen

    # ------------------------------------------------------------------
    # 3) Scoring (KL Similarity + Bigram Hit Rate)
    # ------------------------------------------------------------------
    # KL Score: sum(P_kl * log(P_kl / Q_doc))
    # Since Q_doc is unknown, we treat it as a normalized frequency of words in docs
    # and use a simplified cross-entropy proxy: Score = sum(P_kl * log(P_kl)) - sum(P_kl * log(Q_doc))
    # To make it a similarity, we calculate log-likelihood: Score = sum(P_kl * log(Q_doc))
    # where Q_doc is the empirical distribution of the doc.
    
    final_scores = []
    for i in survivors:
        ws = _words(docs[i])
        hr = order_scores[len(final_scores)]
        
        # KL Score calculation
        doc_counts = Counter(ws)
        doc_total = len(ws)
        kl_score = 0.0
        # We only consider words in the KL vocabulary
        for word, idx in vocab_kl.items():
            p_kl = probs_kl[idx]
            q_doc = doc_counts[word] / doc_total if doc_total > 0 else 0
            if q_doc > 0:
                kl_score += p_kl * np.log(p_kl / q_doc)
            else:
                kl_score += p_kl * np.log(p_kl / 1e-9)
        
        # Normalize KL to be a "similarity" (negative KL is better)
        # We use a simple heuristic: map it to a 0-1 range or use it as a magnitude
        # Given the prompt: "log-likelihood ... relative to target", we'll use the likelihood
        # but we must ensure it's comparable to the bigram hit rate [0, 1].
        # Let's use the log-likelihood sum weighted by P_kl.
        # Because we want higher to be better, we'll use -KL or a transformation.
        # However, standard log-likelihood is negative, so we use the negative of the KL divergence.
        # To keep it simple and robust:
        kl_sim = 1.0 / (1.0 + abs(kl_score)) # Simple mapping to [0, 1]
        
        # Weighted sum: 0.6 KL + 0.4 Bigram
        score = 0.6 * kl_sim + 0.4 * hr
        final_scores.append(score)

    probs = np.array(final_scores)
    
    # ------------------------------------------------------------------
    # 4) Selection
    # ------------------------------------------------------------------
    order = np.argsort(-probs, kind="stable")
    ranked = [survivors[j] for j in order]
    ranked_p = probs[order]
    n_pos = int((ranked_p >= P_KEEP).sum())

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
    n_sel_pos = 0
    n_topup = 0
    done = False
    
    for s in range(0, len(ranked), TOK_BATCH):
        batch = ranked[s : s + TOK_BATCH]
        counts = count_tokens([docs[i] for i in batch])
        for k, (idx, n) in enumerate(zip(batch, counts)):
            r = s + k
            if r < n_pos:
                if tot >= T:
                    done = True; break
                selected.append(idx)
                tot += n
                n_sel_pos += 1
            else:
                if tot >= topup_target:
                    done = True; break
                selected.append(idx)
                tot += n
                n_topup += 1
        if done: break

    if not selected:
        selected = list(survivors)
    else:
        rng.shuffle(selected)
    
    _log(f"Final: {len(selected)} docs, {tot} tokens. Pos: {n_sel_pos}, Topup: {n_topup}")
    return selected
