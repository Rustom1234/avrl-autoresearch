"""Curator: domain classifier + dedup + word-order (fluency) filter
+ paragraph-containment near-duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Drop docs shorter than 200 chars and exact / near duplicates (hash of the
     whitespace-normalised, lowercased first 500 chars; first occurrence kept).
     This is done before vectorising purely to save time; the result is the same.
  2. Char n-gram TF-IDF (char_wb, 1-4, 200k features, sublinear tf) on the head of
     each doc, LogisticRegression(target_sample vs. seeded random pool sample),
     then one self-training round (top 15% vs. bottom 50% of the pool).
  3. Fluency filter (catches word-shuffled docs the char_wb classifier cannot see):
       a. B = lowercased whitespace-token bigrams of all target docs (full text).
       b. For a doc, take ~6000 chars after the header line, split into tokens,
          hit = fraction of adjacent token pairs found in B, computed on the
          first and second half separately; the minimum is used.
       c. Threshold calibrated without labels: B built from the even-indexed
          target docs, statistic computed on the odd-indexed ones;
          thr = 0.6 * 5th percentile.  Docs with min-half hit < thr are dropped.
       d. Second guard: docs with >= 100 tokens whose rate of the function
          bigrams {of the, in the, ", and", ". the", to the} per token is below
          0.25x the target median are dropped.
     If the filter removes more than 70% of the p > 0.5 docs it is considered
     miscalibrated and is disabled.
  4. Rank by score.  Walking the ranking best-first, each fluency-passing doc is
     checked for content-level duplication: its lines of >= 80 normalised chars
     are hashed; dup_frac = char-weighted fraction of those lines already seen
     in accepted docs (or earlier in the same doc).  Docs with >= 3 such lines
     and dup_frac > 0.5 are skipped (their hashes are not recorded).  Accepted
     docs with p > 0.5 are kept up to T tokens; if that yields fewer than 0.5 T
     tokens, top up with the next best accepted docs until 0.5 T.
     If the paragraph step skips more than 40% of the p > 0.5 passing docs it is
     disabled and the selection is redone without it.
  5. Seeded shuffle.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import hashlib
import os
import random
import time

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

HEAD_CHARS = 2000        # chars of each doc fed to the classifier
CHAR_BUDGET = 30_000_000  # total chars vectorised (keeps us inside the time limit)
MIN_CHARS = 200
N_NEG = 3000
TIME_SELF_TRAIN = 105.0  # skip the self-training round if already past this (s)
TIME_FILTER = 140.0      # stop evaluating the fluency / paragraph filters past this (s)
TIME_TOKENISE = 150.0    # switch to len/4 token estimates past this (s)

# fluency filter
FLU_CHARS = 6000
FLU_MIN_TOKENS = 20       # docs with fewer body tokens are not judged
FLU_HALF_MIN_PAIRS = 40   # below this, use the whole-window hit rate
FLU_THR_FACTOR = 0.6
FLU_PCTL = 5
FUNC_MIN_TOKENS = 100
FUNC_FACTOR = 0.25
FLU_MAX_REMOVED = 0.70

# paragraph-containment dedup
PAR_MIN_CHARS = 80        # normalised line length to count as a paragraph
PAR_MIN_LINES = 3         # docs with fewer kept lines are never skipped
PAR_DUP_FRAC = 0.5
PAR_MAX_SKIPPED = 0.40    # safety: disable if more than this share is skipped
PAR_MIN_EVAL = 20         # minimum p > 0.5 docs evaluated before the safety can trip

_FUNC = frozenset([
    ("of", "the"), ("in", "the"), (",", "and"), (".", "the"), ("to", "the"),
])


def _dedup_key(d):
    norm = " ".join(d[:1500].lower().split())[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _body(d):
    """First ~FLU_CHARS chars after the first (header) line."""
    s = d[:FLU_CHARS + 1500].lstrip()
    nl = s.find("\n")
    if nl != -1 and nl <= 200 and len(s) - nl > 200:
        return s[nl + 1: nl + 1 + FLU_CHARS]
    return s[:FLU_CHARS]


def _fluency(d, B):
    """Return (min-half bigram hit rate or None, function-bigram rate, n tokens)."""
    toks = _body(d).lower().split()
    nt = len(toks)
    m = nt - 1
    if m < 1:
        return None, 0.0, nt
    pairs = list(zip(toks, toks[1:]))
    flags = list(map(B.__contains__, pairs))
    if m >= FLU_HALF_MIN_PAIRS:
        h = m // 2
        hit = min(sum(flags[:h]) / h, sum(flags[h:]) / (m - h))
    else:
        hit = sum(flags) / m
    nf = sum(map(_FUNC.__contains__, pairs))
    return hit, nf / nt, nt


def _bigrams(texts):
    B = set()
    for t in texts:
        toks = t.lower().split()
        B.update(zip(toks, toks[1:]))
    return B


def _par_check(d, seen_par):
    """Return (is_duplicate, set of paragraph hashes of d)."""
    local = set()
    n_kept = 0
    tot = 0
    dup = 0
    for line in d.split("\n"):
        if len(line) < PAR_MIN_CHARS:
            continue  # normalising never lengthens a line
        norm = " ".join(line.lower().split())
        ln = len(norm)
        if ln < PAR_MIN_CHARS:
            continue
        h = hash(norm)
        n_kept += 1
        tot += ln
        if h in local or h in seen_par:
            dup += ln
        local.add(h)
    is_dup = n_kept >= PAR_MIN_LINES and tot > 0 and (dup / tot) > PAR_DUP_FRAC
    return is_dup, local


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)

    def fallback():
        idx = list(range(n_docs))
        random.Random(seed).shuffle(idx)
        return idx

    if n_docs == 0:
        return []

    # ---- 1. length filter + dedup (first occurrence wins) ------------------
    seen = set()
    cand = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        cand.append(i)
    del seen
    if not cand or not target_sample:
        return fallback()
    n = len(cand)

    # ---- 2. classifier ---------------------------------------------------
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        head = int(max(300, min(HEAD_CHARS, CHAR_BUDGET // (n + len(target_sample)))))
        texts = [docs[i][:head] for i in cand] + [str(t)[:head] for t in target_sample]
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 4),
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X = vec.fit_transform(texts).tocsr()
        del texts
        Xp = X[:n]
        Xt = X[n:]
        n_t = Xt.shape[0]

        from scipy.sparse import vstack

        neg_rows = rng.sample(range(n), min(N_NEG, n))
        Xtr = vstack([Xt, Xp[neg_rows]]).tocsr()
        ytr = np.concatenate([np.ones(n_t, dtype=np.int32),
                              np.zeros(len(neg_rows), dtype=np.int32)])
        clf = LogisticRegression(class_weight="balanced", C=10, max_iter=200)
        clf.fit(Xtr, ytr)
        scores = clf.predict_proba(Xp)[:, 1]

        # one self-training round
        if time.time() - t0 < TIME_SELF_TRAIN and n >= 20:
            order = np.argsort(-scores, kind="stable")
            n_pos = max(1, int(0.15 * n))
            n_neg = max(1, int(0.50 * n))
            pos_rows = order[:n_pos].tolist()
            neg_rows2 = order[n - n_neg:].tolist()
            if len(pos_rows) > 8000:
                pos_rows = rng.sample(pos_rows, 8000)
            if len(neg_rows2) > 16000:
                neg_rows2 = rng.sample(neg_rows2, 16000)
            Xtr2 = vstack([Xt, Xp[pos_rows], Xp[neg_rows2]]).tocsr()
            ytr2 = np.concatenate([
                np.ones(n_t + len(pos_rows), dtype=np.int32),
                np.zeros(len(neg_rows2), dtype=np.int32),
            ])
            try:
                clf2 = LogisticRegression(class_weight="balanced", C=10, max_iter=100)
                clf2.fit(Xtr2, ytr2)
                scores = clf2.predict_proba(Xp)[:, 1]
            except Exception:
                pass
        scores = np.asarray(scores, dtype=np.float64)
        del X, Xp, Xt
    except Exception:
        return fallback()

    order = np.argsort(-scores, kind="stable")

    # ---- 3. fluency (word-order) filter ------------------------------------
    filter_on = False
    B = set()
    hit_thr = 0.0
    func_thr = 0.0
    status = np.full(n, -1, dtype=np.int8)  # -1 unknown, 0 fail, 1 pass
    try:
        tgt = [str(t) for t in target_sample]
        B = _bigrams(tgt)

        # calibration: bigrams from even-indexed targets, scored on odd-indexed
        B_even = _bigrams(tgt[0::2])
        held = []
        for t in tgt[1::2]:
            hit, _, nt = _fluency(t, B_even)
            if hit is not None and nt >= FLU_MIN_TOKENS:
                held.append(hit)
        del B_even
        if len(held) >= 10:
            hit_thr = FLU_THR_FACTOR * float(np.percentile(held, FLU_PCTL))

        frates = []
        for t in tgt:
            _, fr, nt = _fluency(t, B)
            if nt >= FUNC_MIN_TOKENS:
                frates.append(fr)
        if len(frates) >= 10:
            func_thr = FUNC_FACTOR * float(np.median(frates))

        filter_on = bool(B) and (hit_thr > 0.0 or func_thr > 0.0)
    except Exception:
        filter_on = False

    def passes(r):
        """Does candidate row r pass the fluency filter? (cached)"""
        if not filter_on:
            return True
        s = status[r]
        if s >= 0:
            return bool(s)
        if time.time() - t0 > TIME_FILTER:
            return True  # out of time: unknown docs are kept
        try:
            hit, fr, nt = _fluency(docs[cand[r]], B)
            ok = True
            if hit is not None and nt >= FLU_MIN_TOKENS:
                if hit_thr > 0.0 and hit < hit_thr:
                    ok = False
                elif func_thr > 0.0 and nt >= FUNC_MIN_TOKENS and fr < func_thr:
                    ok = False
        except Exception:
            ok = True
        status[r] = 1 if ok else 0
        return ok

    if filter_on:
        try:
            n_hi = int(np.sum(scores > 0.5))
            # docs used for the miscalibration check
            n_check = n_hi if n_hi >= 50 else min(n, max(n_hi, 2000))
            check_rows = order[:n_check].tolist()
            for r in check_rows:
                if time.time() - t0 > TIME_FILTER:
                    break
                passes(r)
            st = status[check_rows]
            n_eval = int(np.sum(st >= 0))
            n_removed = int(np.sum(st == 0))
            if n_eval > 0 and n_removed / n_eval > FLU_MAX_REMOVED:
                filter_on = False  # miscalibrated
        except Exception:
            filter_on = False

    # ---- 4. rank, paragraph dedup, count tokens lazily, cut -----------------
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    def count_tokens(texts):
        if not texts:
            return []
        if enc is not None and time.time() - t0 < TIME_TOKENISE:
            try:
                return [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                pass
        return [max(1, len(x) // 4) for x in texts]

    tok_cache = {}  # candidate row -> token count

    def run_selection(use_par):
        """Return (selected, n p>0.5 docs par-checked, n p>0.5 docs par-skipped)."""
        selected = []
        total = 0
        half_T = 0.5 * T
        CHUNK = 1000
        done = False
        seen_par = set()
        hi_eval = 0
        hi_skip = 0
        for start in range(0, n, CHUNK):
            rows = [int(r) for r in order[start:start + CHUNK]]
            rows = [r for r in rows if passes(r)]

            if use_par:
                acc = []
                for r in rows:
                    if time.time() - t0 > TIME_FILTER:
                        acc.append(r)  # out of time: accept without checking
                        continue
                    try:
                        is_dup, hs = _par_check(docs[cand[r]], seen_par)
                    except Exception:
                        acc.append(r)
                        continue
                    hi = scores[r] > 0.5
                    if hi:
                        hi_eval += 1
                    if is_dup:
                        if hi:
                            hi_skip += 1
                        continue
                    seen_par.update(hs)
                    acc.append(r)
                rows = acc

            missing = [r for r in rows if r not in tok_cache]
            if missing:
                cnts = count_tokens([docs[cand[r]] for r in missing])
                for r, c in zip(missing, cnts):
                    tok_cache[r] = c

            for r in rows:
                if scores[r] > 0.5:
                    if total >= T:
                        done = True
                        break
                else:
                    if total >= half_T:
                        done = True
                        break
                selected.append(cand[r])
                total += tok_cache[r]
            if done:
                break
        return selected, hi_eval, hi_skip

    try:
        selected, hi_eval, hi_skip = run_selection(True)
        if hi_eval >= PAR_MIN_EVAL and hi_skip / hi_eval > PAR_MAX_SKIPPED:
            # paragraph dedup looks miscalibrated: fall back to previous behaviour
            selected, _, _ = run_selection(False)
    except Exception:
        try:
            selected, _, _ = run_selection(False)
        except Exception:
            return fallback()

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
