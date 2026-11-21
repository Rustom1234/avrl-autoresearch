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
MAX_FEATURES = 50_000      # reduced for unigram count
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

    wc = Counter()
    for t in target_sample:
        wc.update(_all_words(t))
    
    # Build vocab for bigrams and also for our unigram count
    # Keep common words from target_sample
    vocab_list = [w for w, _ in wc.most_common(TOP_WORDS)]
    vocab = {w: i for i, w in enumerate(vocab_list)}
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
        _log(f"bigram calibration p2={np.percentile(cal, 2):.3f} -> cutoff={cutoff:.3f}")
    
    table = _build_table(target_sample, vocab, V)

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
        if len(ws) < 20 or (len(ws) >= 50 and len(set(ws)) / len(ws) < MIN_UNIQ_RATIO): continue
        
        hr = _hit_rate(ws, vocab, V, table)
        order_scores.append(hr)
        if hr < cutoff: continue
        survivors.append(i)

    if not survivors:
        idx = list(range(n_docs))
        rng.shuffle(idx)
        return idx

    # --- Unigram Count Vectorizer & Logistic Regression ---
    # Filter vocab to only include words that appear in the target_sample significantly
    # To keep it simple and fast, we use words that are in our vocab list.
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        pos = []
        for t in target_sample:
            pos.extend(_chunks(t))
        if len(pos) > MAX_POS:
            pos = rng.sample(pos, MAX_POS)
            
        neg_ids = rng.sample(survivors, min(N_NEG, len(survivors)))
        neg = [docs[i][:CHUNK_CHARS] for i in neg_ids]
        
        vec = CountVectorizer(
            vocabulary=vocab,
            min_df=2,
            dtype=np.float32,
            token_pattern=r"\S+"
        )
        Xtr = vec.fit_transform(pos + neg)
        ytr = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
        
        clf = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)

        # Predict probabilities
        clip = int(max(600, min(CHUNK_CHARS, SCORE_CHAR_BUDGET // len(survivors))))
        out = []
        B = 4000
        for s in range(0, len(survivors), B):
            batch = [docs[i][:clip] for i in survivors[s : s + B]]
            out.append(clf.predict_proba(vec.transform(batch))[:, 1])
        probs = np.concatenate(out)

    except Exception as e:
        _log(f"Classifier failed ({type(e).__name__}), using bigram hit rates.")
        probs = np.array(order_scores)
        # Normalize to 0-1
        if probs.max() > probs.min():
            probs = (probs - probs.min()) / (probs.max() - probs.min())
        else:
            probs = np.zeros_like(probs)

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
    pos_tokens = 0
    n_sel_pos = 0
    n_topup = 0
    done = False
    
    for s in range(0, len(ranked), TOK_BATCH):
        batch_idx = ranked[s : s + TOK_BATCH]
        counts = count_tokens([docs[i] for i in batch_idx])
        for k, (idx, n) in enumerate(zip(batch_idx, counts)):
            r = s + k
            if r < n_pos:
                if tot >= T:
                    done = True
                    break
                selected.append(idx)
                tot += n
                pos_tokens += n
                n_sel_pos += 1
            else:
                if tot >= topup_target:
                    done = True
                    break
                selected.append(idx)
                tot += n
                n_topup += 1
        if done: break

    if not selected:
        selected = list(survivors)
    rng.shuffle(selected)
    _log(f"Final: {len(selected)} docs, {tot} tokens ({tot/T:.2f} T)")
    return selected
