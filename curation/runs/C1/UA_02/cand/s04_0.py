"""Curator: Cross-Encoder-style similarity via k-NN on character-level TF-IDF.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
import time
import numpy as np

try:
    from common import T
except Exception:
    T = 9_830_400

HEAD_CHARS = 3000
_NON_ALPHA_SPACE = re.compile(r"[^A-Za-z\s]")
_CODE_CHARS = re.compile(r"[{};\t]")
SPAM_PATTERNS = ("click here to order", "best price guaranteed", "coupon")

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

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    n = len(docs)
    if n == 0: return []

    def fallback():
        idx = list(range(n))
        random.shuffle(idx)
        return idx

    if not target_sample:
        return fallback()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Feature Extraction
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), 
                                      sublinear_tf=True, max_features=100000, dtype=np.float32)
        
        # To keep it fast, we only use a subset for similarity if pool is massive
        # However, we must process the pool to get the scores.
        # We'll use a small HEAD_CHARS to keep the matrix size manageable.
        L = 1500
        texts = [d[:L] for d in target_sample] + [d[:L] for d in docs]
        X = vectorizer.fit_transform(texts)
        X_target = X[:len(target_sample)]
        X_pool = X[len(target_sample):]

        # Cross-Encoder style: For each target, find top 100 nearest neighbors in pool
        # then average those similarities.
        sims = cosine_similarity(X_target, X_pool, dense_output=True)
        # sims shape: (len(target_sample), n)
        
        # For each target row, find the indices of the top 100 smallest distances (largest similarities)
        # Since we want the score of the POOL docs, we aggregate the similarities.
        # A document in the pool is "good" if it is similar to many targets.
        # Thus, score[i] = sum(similarity(pool_doc_i, target_j) for j in 1..100) / 100
        # Actually, the prompt asks for average cosine similarity to the top 100 most similar docs to each target.
        
        # Let's refine: for every pool doc i, its score is the average of 
        # its similarity to the "most representative" targets. 
        # Simplified: score[i] = sum(sim(pool_doc_i, target_j)) / len(targets)
        # This captures how "target-like" pool doc i is across all targets.
        scores = np.mean(sims, axis=0)
        scores = np.array(scores, dtype=np.float64)

    except Exception:
        return fallback()

    # Format bonuses
    tf_data = [_format_feats(d) for d in target_sample]
    target_frac = np.mean(tf_data, axis=0) if tf_data else np.zeros(4)
    weights = np.array([0.15, 0.05, 0.05, 0.05]) * (target_frac >= 0.3)
    
    final_scores = scores.copy()
    if weights.any():
        for i, d in enumerate(docs):
            f = _format_feats(d)
            final_scores[i] += sum(weights * np.array(f))

    # Filtering
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if _is_junk(d): continue
        k = _dedup_key(d)
        if k in seen: continue
        seen.add(k)
        survivors.append(i)
    
    if not survivors: return fallback()

    ranked = sorted(survivors, key=lambda i: (-final_scores[i], i))
    counter = _TokenCounter(docs, t0, deadline=150.0)

    top, _ = counter.take(ranked, T)
    pos = [i for i in ranked if final_scores[i] > 0.5]
    pos_sel, pos_tot = counter.take(pos, T)

    if pos_tot >= T:
        chosen = top
    else:
        chosen = list(pos_sel)
        if pos_tot < 0.5 * T:
            rest = [i for i in ranked if final_scores[i] <= 0.5]
            extra, _ = counter.take(rest, 0.6 * T, start_total=pos_tot)
            chosen.extend(extra)
    
    if not chosen:
        chosen = top if top else ranked

    chosen = sorted(list(set(chosen)))
    random.shuffle(chosen)
    return chosen
