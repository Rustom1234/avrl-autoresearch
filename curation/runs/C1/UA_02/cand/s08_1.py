"""Curator: Multi-Annotated similarity filter + format heuristics + junk hard-drops + near-duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens; a list with fewer than T tokens is cycled.
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

HEAD_CHARS = 3000
SPAM_PATTERNS = ("click here to order", "best price guaranteed", "coupon")
_NON_ALPHA_SPACE = re.compile(r"[^A-Za-z\s]")
_CODE_CHARS = re.compile(r"[{};\t]")

def _has_header(d):
    s = d.lstrip()
    if not s.startswith("= "):
        return False
    first = s.split("\n", 1)[0].rstrip()
    return len(first) > 3 and first.endswith("=")

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

def _get_quality_score(d):
    words = d.lower().split()
    if not words: return 0.1
    u_ratio = len(set(words)) / len(words)
    len_score = min(len(d) / 2000.0, 1.0)
    header_bonus = 1.5 if _has_header(d) else 1.0
    return (0.4 * len_score + 0.6 * u_ratio) * header_bonus

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0: return []
    
    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    if not target_sample: return fallback()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), 
                                    sublinear_tf=True, max_features=50000, dtype=np.float32)
        
        all_texts = target_sample + docs
        X = char_vec.fit_transform(all_texts)
        X_target = X[:len(target_sample)]
        X_pool = X[len(target_sample):]

        q_scores = np.array([_get_quality_score(d) for d in target_sample])
        q_norm = q_scores / (np.sum(q_scores) + 1e-9)
        sim_matrix = X_pool.dot(X_target.T).toarray()
        scores = np.dot(sim_matrix, q_norm)

    except Exception:
        return fallback()

    seen = set()
    survivors = []
    scores_list = []
    for i in range(n):
        d = docs[i]
        if _is_junk(d): continue
        k = _dedup_key(d)
        if k in seen: continue
        seen.add(k)
        survivors.append(i)
        scores_list.append(scores[i])

    if not survivors: return fallback()

    scores_array = np.array(scores_list)
    target_feats = np.array([_format_feats(d) for d in target_sample])
    target_frac = target_feats.mean(axis=0)
    weights = np.array([0.2, 0.1, 0.1, 0.1]) * (target_frac >= 0.3)
    
    final_scores = scores_array.copy()
    for idx, i in enumerate(survivors):
        f = np.array([_format_feats(docs[i])])
        final_scores[idx] += np.sum(weights * f)

    ranked = sorted(survivors, key=lambda i: (-final_scores[survivors.index(i)], i))
    
    counter = _TokenCounter(docs, t0, deadline=150.0)
    top, _ = counter.take(ranked, T)

    pos_indices = [i for i in ranked if scores[survivors.index(i)] > 0.1]
    pos_sel, pos_tot = counter.take(pos_indices, T)

    if pos_tot >= T:
        chosen = top
    else:
        chosen = list(pos_sel)
        if pos_tot < 0.5 * T:
            rest = [i for i in ranked if scores[survivors.index(i)] <= 0.1]
            extra, _ = counter.take(rest, 0.6 * T, start_total=pos_tot)
            chosen.extend(extra)
        else:
            chosen = list(pos_sel)
    
    chosen = sorted(list(set(chosen)))
    random.Random(seed).shuffle(chosen)
    return chosen
