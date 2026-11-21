"""Curator: target-similarity filter (PU logistic regression on TF-IDF) + format
heuristics + junk hard-drops + near-duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import time
import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

HEAD_CHARS = 3000          # chars of each doc used for features (may shrink for speed)
N_NEG = 3000               # random pool docs used as (noisy) negatives
SPAM_PATTERNS = ("click here to order", "best price guaranteed", "coupon")
_NON_ALPHA_SPACE = re.compile(r"[^A-Za-z\s]")
_CODE_CHARS = re.compile(r"[{};\t]")

# --------------------------------------------------------------------------- #
# cheap per-document heuristics
# --------------------------------------------------------------------------- #
def _has_header(d):
    s = d.lstrip()
    if not s.startswith("= "):
        return False
    first = s.split("\n", 1)[0].rstrip()
    return len(first) > 3 and first.endswith(" =")

def _format_feats(d):
    head = d[:HEAD_CHARS]
    return (_has_header(d), " @-@ " in head, " , " in head, " . " in head)

def _is_junk(d):
    if len(d.strip()) < 200:
        return True
    low = d.lower()
    for p in SPAM_PATTERNS:
        if p in low:
            return True
    head = d[:HEAD_CHARS]
    n = len(head)
    if n == 0:
        return True
    # non-ASCII fraction
    non_ascii = n - len(head.encode("ascii", "ignore"))
    if non_ascii / n > 0.20:
        return True
    # symbol / digit heavy text (fraction of ASCII letters + whitespace is low)
    alpha_space = len(_NON_ALPHA_SPACE.sub("", head))
    if alpha_space / n < 0.60:
        return True
    # code-like characters
    if len(_CODE_CHARS.findall(head)) / n > 0.015:
        return True
    # repetitive text: low unique-word ratio on the first 400 words
    words = head.lower().split()[:400]
    if len(words) >= 50 and len(set(words)) / len(words) < 0.30:
        return True
    return False

def _dedup_key(d):
    return hash(" ".join(d[:2500].lower().split())[:500])

# --------------------------------------------------------------------------- #
# token counting (lazy, cached)
# --------------------------------------------------------------------------- #
class _TokenCounter:
    def __init__(self, docs, t0, deadline):
        self.docs = docs
        self.cache = {}
        self.t0 = t0
        self.deadline = deadline
        self.chars_seen = 0
        self.toks_seen = 0
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None

    def _estimate(self, d):
        ratio = (self.chars_seen / self.toks_seen) if self.toks_seen > 0 else 4.0
        return int(len(d) / max(ratio, 1.0)) + 1

    def count(self, idxs):
        missing = [i for i in idxs if i not in self.cache]
        if missing:
            use_enc = self.enc is not None and (time.time() - self.t0) < self.deadline
            if use_enc:
                try:
                    texts = [self.docs[i] for i in missing]
                    toks = self.enc.encode_ordinary_batch(texts, num_threads=8)
                    for i, d, t in zip(missing, texts, toks):
                        self.cache[i] = len(t)
                        self.chars_seen += len(d)
                        self.toks_seen += len(t)
                except Exception:
                    use_enc = False
            if not use_enc:
                for i in missing:
                    if i not in self.cache:
                        self.cache[i] = self._estimate(self.docs[i])
        return [self.cache[i] for i in idxs]

    def take(self, seq, target, start_total=0):
        out, tot = [], start_total
        if tot >= target:
            return out, tot
        for s in range(0, len(seq), 256):
            chunk = seq[s:s + 256]
            for i, c in zip(chunk, self.count(chunk)):
                out.append(i)
                tot += c
                if tot >= target:
                    return out, tot
        return out, tot

# --------------------------------------------------------------------------- #
# similarity model
# --------------------------------------------------------------------------- #
def _pick_head_len(docs, char_vec, budget_s=55.0):
    n = len(docs)
    lens = np.fromiter((len(d) for d in docs), dtype=np.int64, count=n)
    analyzer = char_vec.build_analyzer()
    step = max(1, n // 200)
    sample = [d[:HEAD_CHARS] for d in docs[::step][:200]]
    chars = sum(len(s) for s in sample)
    t = time.time()
    for s in sample:
        analyzer(s)
    dt = max(time.time() - t, 1e-4)
    rate = max(chars, 1) / dt  # chars / second for analysis alone
    for L in (3000, 2500, 2000, 1500, 1200, 1000, 800, 600, 450, 300, 200, 150):
        est = 2.5 * float(np.minimum(lens, L).sum()) / rate
        if est <= budget_s:
            return L
    return 150

def _target_probs(docs, target_sample, seed, t0):
    from scipy.sparse import hstack, vstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    nt, n = len(target_sample), len(docs)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                               max_features=200_000, dtype=np.float32)
    L = _pick_head_len(docs, char_vec)
    texts = [d[:L] for d in target_sample] + [d[:L] for d in docs]

    mats = []
    try:
        mats.append(char_vec.fit_transform(texts))
    except ValueError:
        pass
    
    if not mats or (time.time() - t0) < 105:
        try:
            word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                       max_features=200_000, min_df=2, dtype=np.float32)
            mats.append(word_vec.fit_transform(texts))
        except ValueError:
            pass
    
    if not mats:
        return np.full(n, 0.5)
    X = hstack(mats).tocsr() if len(mats) > 1 else mats[0].tocsr()

    # --- Improved PU-learning: Hard filtering of negatives ---
    # Filter out junk/noise for the negative pool selection
    clean_pool_indices = []
    for i in range(nt, n):
        if not _is_junk(docs[i]):
            clean_pool_indices.append(i)
    
    rng = random.Random(seed)
    neg_count = min(N_NEG, len(clean_pool_indices))
    neg_indices = rng.sample(clean_pool_indices, neg_count)
    
    Xp = X[:nt]
    Xpool = X[nt:]
    # Re-index Xpool to select from the filtered clean set
    X_neg = Xpool[neg_indices]
    Xtr = vstack([Xp, X_neg]).tocsr()
    ytr = np.concatenate([np.ones(nt, dtype=np.int32), np.zeros(len(neg_indices), dtype=np.int32)])
    
    # Higher class weight for target_sample to compensate for smaller positive set
    # Using a high weight (e.g., 10.0) since nt (200) is small compared to N_NEG (3000)
    clf = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 10.0}, solver="liblinear", max_iter=1000)
    clf.fit(Xtr, ytr)
    
    return clf.predict_proba(Xpool)[:, 1]

# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    if not target_sample:
        return fallback()

    try:
        prob = np.asarray(_target_probs(docs, target_sample, seed, t0), dtype=np.float64)
    except Exception:
        return fallback()

    tf = np.array([_format_feats(d) for d in target_sample], dtype=np.float64)
    target_frac = tf.mean(axis=0) if len(tf) else np.zeros(4)
    weights = np.array([0.15, 0.05, 0.05, 0.05]) * (target_frac >= 0.3)
    score = prob.copy()
    if weights.any():
        for i, d in enumerate(docs):
            f = _format_feats(d)
            score[i] += (weights[0] * f[0] + weights[1] * f[1]
                         + weights[2] * f[2] + weights[3] * f[3])

    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if _is_junk(d):
            continue
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        survivors.append(i)
    if not survivors:
        return fallback()

    ranked = sorted(survivors, key=lambda i: (-score[i], i))
    counter = _TokenCounter(docs, t0, deadline=150.0)

    top, _ = counter.take(ranked, T)
    pos = [i for i in ranked if prob[i] > 0.5]
    pos_sel, pos_tot = counter.take(pos, T)

    if pos_tot >= T:
        chosen = top
    else:
        chosen = list(pos_sel)
        if pos_tot < 0.5 * T:
            rest = [i for i in ranked if prob[i] <= 0.5]
            extra, _ = counter.take(rest, 0.6 * T, start_total=pos_tot)
            chosen.extend(extra)
    if not chosen:
        chosen = top if top else ranked

    chosen = sorted(set(chosen))
    random.Random(seed).shuffle(chosen)
    return chosen
