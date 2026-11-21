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
WORDS_CAP = 2000           # words per doc inspected by the bigram filter
MIN_BIGRAMS = 10           # docs with fewer in-vocab bigrams get hit rate 0
CAL_PERCENTILE = 2.0       # percentile of held-out target hit rates
CAL_SLACK = 0.9            # multiplier on that percentile
CHUNK_CHARS = 2000         # positive chunk size / negative truncation
N_NEG = 3000               # pool docs sampled as negatives
MAX_POS = 6000             # cap on positive chunks
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
    """Flat codes a*V+b for adjacent word pairs where both words in vocab."""
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
    # Bigram reference built from the target sample
    # ------------------------------------------------------------------
    wc = Counter()
    for t in target_sample:
        wc.update(_all_words(t))
    vocab = {w: i for i, (w, _) in enumerate(wc.most_common(TOP_WORDS))}
    V = max(len(vocab), 1)

    cutoff = 0.0
    half = len(target_sample) // 2
    if half >= 2:
        cal_table = _build_table(target_sample[:half], vocab, V)
        cal_scores = [
            _hit_rate(_words(t), vocab, V, cal_table) for t in target_sample[half:]
        ]
        cal = np.asarray(cal_scores, dtype=np.float64)
        cutoff = float(np.percentile(cal, CAL_PERCENTILE)) * CAL_SLACK
        _log(
            "bigram calibration (held-out target): "
            f"min={cal.min():.3f} p2={np.percentile(cal, 2):.3f} "
            f"p10={np.percentile(cal, 10):.3f} median={np.median(cal):.3f} "
            f"max={cal.max():.3f} -> cutoff={cutoff:.3f}"
        )
    table = _build_table(target_sample, vocab, V)

    # ------------------------------------------------------------------
    # 1) dedupe, 2) junk guards, 3) word-order filter (single pass)
    # ------------------------------------------------------------------
    seen = set()
    n_dup = n_short = n_alpha = n_rep = n_order = 0
    survivors = []
    order_scores = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            n_short += 1
            continue
        L = len(d)
        head = _WS.sub(" ", d[: DEDUPE_PREFIX * 2].lower()).strip()[:DEDUPE_PREFIX]
        key = (head, L.bit_length())
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        if L < MIN_CHARS:
            n_short += 1
            continue
        sample = d[:5000]
        non_alpha = len(_ALPHA_SPACE.sub("", sample))
        if 1.0 - non_alpha / len(sample) < MIN_ALPHA_FRAC:
            n_alpha += 1
            continue
        ws = _words(d)
        if len(ws) < 20 or (len(ws) >= 50 and len(set(ws)) / len(ws) < MIN_UNIQ_RATIO):
            n_rep += 1
            continue
        hr = _hit_rate(ws, vocab, V, table)
        order_scores.append(hr)
        if hr < cutoff:
            n_order += 1
            continue
        survivors.append(i)
    del seen

    _log(f"dropped duplicates: {n_dup}")
    _log(f"dropped short (<{MIN_CHARS} chars): {n_short}")
    _log(f"dropped low alpha fraction (<{MIN_ALPHA_FRAC}): {n_alpha}")
    _log(f"dropped repetitive / too few words: {n_rep}")
    if order_scores:
        os_ = np.asarray(order_scores)
        _log(
            "pool bigram hit rate: "
            + " ".join(
                f"p{q}={np.percentile(os_, q):.3f}" for q in (5, 25, 50, 75, 95)
            )
        )
    _log(f"dropped by word-order filter (<{cutoff:.3f}): {n_order}")
    _log(f"survivors after filters: {len(survivors)}  ({time.time() - t0:.1f}s)")

    if not survivors:
        _log("no survivors; falling back to seeded random order of the whole pool")
        idx = list(range(n_docs))
        rng.shuffle(idx)
        return idx

    # ------------------------------------------------------------------
    # 4) Coherence Scoring
    # ------------------------------------------------------------------
    _log("Calculating Coherence Scores...")
    probs = np.zeros(len(survivors), dtype=np.float64)
    
    def calculate_coherence(text, vocab, V, table):
        ws = _all_words(text)
        if len(ws) < 2: return 0.0
        ids = np.fromiter(map(vocab.get, ws, repeat(-1)), dtype=np.int64, count=len(ws))
        a = ids[:-1]
        b = ids[1:]
        m = (a >= 0) & (b >= 0)
        if not np.any(m): return 0.0
        
        codes = a[m] * V + b[m]
        # Weighted sum: higher weight to earlier bigrams (decaying weight)
        # Weights are roughly 1.0 down to 0.1 or similar.
        # Here we use a linear decay based on relative position of the bigram in the doc.
        indices = np.where(m)[0]
        # Position in sequence of bigrams
        rel_pos = np.arange(len(indices))
        max_pos_score = len(indices)
        # Weighting factor: 1.0 at start, decreasing to 0.1 at end
        weights = np.linspace(1.0, 0.1, max_pos_score).tolist() if max_pos_score > 0 else [0]
        
        hit_vec = table[codes]
        weighted_hits = np.dot(hit_vec, weights)
        # Normalize by total possible bigrams to get a 0-1ish score
        return float(weighted_hits / (len(indices) * 1.0)) if len(indices) > 0 else 0.0

    # Apply scoring
    for idx, doc_idx in enumerate(survivors):
        probs[idx] = calculate_coherence(docs[doc_idx], vocab, V, table)

    _log(
        f"coherence scores: "
        + " ".join(
            f"p{q}={np.percentile(probs, q):.3f}" for q in (5, 25, 50, 75, 95)
        )
        + f"  ({time.time() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 5) selection by score with token budget
    # ------------------------------------------------------------------
    order = np.argsort(-probs, kind="stable")
    ranked = [survivors[j] for j in order]
    ranked_p = probs[order]
    n_pos = int((ranked_p >= P_KEEP).sum())
    _log(f"survivors with p >= {P_KEEP}: {n_pos}")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count_tokens(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
    except Exception as e:
        _log(f"tiktoken unavailable ({e}); estimating tokens as chars/4")
        def count_tokens(texts):
            return [max(1, len(x) // 4) for x in texts]

    topup_target = int(TOPUP_FRAC * T)
    selected = []
    tot = 0
    pos_tokens = 0
    n_sel_pos = 0
    n_topup = 0
    done = False
    for s in range(0, len(ranked), TOK_BATCH):
        batch = ranked[s : s + TOK_BATCH]
        counts = count_tokens([docs[i] for i in batch])
        for k, (i, n) in enumerate(zip(batch, counts)):
            r = s + k
            if r < n_pos:
                if tot >= T:
                    done = True
                    break
                selected.append(i)
                tot += n
                pos_tokens += n
                n_sel_pos += 1
            else:
                if tot >= topup_target:
                    done = True
                    break
                selected.append(i)
                tot += n
                n_topup += 1
        if done:
            break

    _log(
        f"selected p>={P_KEEP}: {n_sel_pos} docs, {pos_tokens} tokens "
        f"({pos_tokens / T:.2f} T)"
    )
    _log(f"top-up docs (p<{P_KEEP}): {n_topup}")
    if selected:
        _log(f"lowest selected score: {float(ranked_p[len(selected) - 1]):.3f}")
    _log(
        f"final: {len(selected)} docs, {tot} tokens ({tot / T:.2f} T)"
        f"{' -> trainer will cycle' if tot < T else ''}  ({time.time() - t0:.1f}s)"
    )

    if not selected:
        selected = list(survivors)
    rng.shuffle(selected)
    return selected
