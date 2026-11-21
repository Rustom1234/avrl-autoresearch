"""Curator: k-NN target-similarity filter (Word + Char TF-IDF) + format
heuristics + junk hard-drops + near-duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
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

HEAD_CHARS = 3000          # chars of each doc used for features
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
    non_ascii = n - len(head.encode("ascii", "ignore"))
    if non_ascii / n > 0.20:
        return True
    alpha_space = len(_NON_ALPHA_SPACE.sub("", head))
    if alpha_space / n < 0.60:
        return True
    if len(_CODE_CHARS.findall(head)) / n > 0.015:
        return True
    words = head.lower().split()[:400]
    if len(words) >= 50 and len(set(words)) / len(words) < 0.30:
        return True
    return False

def _dedup_key(d):
    return hash(" ".join(d[:2500].lower().split())[:500])

# --------------------------------------------------------------------------- #
# token counting
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
# k-NN similarity model
# --------------------------------------------------------------------------- #
def _get_knn_scores(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    nt = len(target_sample)
    n = len(docs)
    if nt == 0: return np.full(n, 0.5)

    # Build Word and Char TF-IDF
    # We use a limited max_features to ensure we stay within memory/time constraints
    v_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)

    # Fit on both targets and pool
    texts = target_sample + docs
    w_mats = v_word.fit_transform(texts)
    c_mats = v_char.fit_transform(texts)
    
    # Combine mats (stacking results of both features)
    # We convert to dense/sparse combination manually to save memory
    # Score = 0.5 * Word_sim + 0.5 * Char_sim
    w_sim = cosine_similarity(w_mats[nt:], w_mats[:nt])
    c_sim = cosine_similarity(c_mats[nt:], c_mats[:nt])
    
    # Average similarities (k=5)
    # w_sim shape: (n, nt)
    # find 5 nearest neighbors in target_sample for each pool doc
    # Since we want average of k-NN, we average the sims for the top 5 smallest distances
    # However, cosine_similarity provides high values for similar items. 
    # We want the top 5 highest cosine similarities.
    combined_sim = (w_sim + c_sim) / 2.0
    
    # Get top 5 indices for each pool document
    # We use argpartition for efficiency: O(n)
    k = min(5, nt)
    # Find indices of k largest similarities
    # combined_sim is (n, nt)
    # We want to average the k largest values in each row
    # For small n, we can just sort. For large n, argpartition is better.
    scores = np.zeros(n)
    for i in range(n):
        # Get the top k similarity values
        top_k_vals = np.sort(combined_sim[i])[-k:]
        scores[i] = np.mean(top_k_vals)
    
    return scores

# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0: return []

    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    if not target_sample:
        return fallback()

    try:
        scores = _get_knn_scores(docs, target_sample, seed)
    except Exception:
        return fallback()

    # Format bonuses
    tf = np.array([_format_feats(d) for d in target_sample], dtype=np.float64)
    target_frac = tf.mean(axis=0) if len(tf) else np.zeros(4)
    weights = np.array([0.15, 0.05, 0.05, 0.05]) * (target_frac >= 0.3)
    
    final_scores = scores.copy()
    if weights.any():
        for i in range(n):
            f = _format_feats(docs[i])
            final_scores[i] += (weights[0] * f[0] + weights[1] * f[1]
                                 + weights[2] * f[2] + weights[3] * f[3])

    # Filtering
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

    # Rank survivors
    ranked = sorted(survivors, key=lambda i: (-final_scores[i], i))
    counter = _TokenCounter(docs, t0, deadline=160.0)

    # Strategy: try to fill T with target-like docs (score > 0.5 is arbitrary but safe)
    # k-NN scores are averages of similarities, so 0.5 is a reasonable threshold for "related"
    pos_indices = [i for i in ranked if final_scores[i] > 0.4]
    pos_sel, pos_tot = counter.take(pos_indices, T)

    if pos_tot >= T:
        chosen = ranked[:len(pos_sel)] # This logic is slightly off, just use the take results
        chosen = pos_sel
    else:
        chosen = list(pos_sel)
        # Fill the rest with the highest scoring docs regardless of threshold
        remaining_indices = [i for i in ranked if i not in pos_sel]
        extra, _ = counter.take(remaining_indices, T - pos_tot, start_total=pos_tot)
        chosen.extend(extra)

    # Final cleanup
    chosen = sorted(list(set(chosen)))
    random.Random(seed).shuffle(chosen)
    return chosen
