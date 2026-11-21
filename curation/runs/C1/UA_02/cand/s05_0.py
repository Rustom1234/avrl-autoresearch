"""Curator: Centroid-based similarity filter using character-level TF-IDF.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
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
_NON_ALPHA_SPACE = re.compile(r"[^A-Za-z\s]")
_CODE_CHARS = re.compile(r"[{};\t]")
SPAM_PATTERNS = ("click here to order", "best price guaranteed", "coupon")

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
    n = len(docs)
    if n == 0: return []
    
    rng = random.Random(seed)
    
    # Filter junk and dedup
    survivors = []
    seen = set()
    for i, d in enumerate(docs):
        if _is_junk(d): continue
        k = _dedup_key(d)
        if k in seen: continue
        seen.add(k)
        survivors.append(i)
    
    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Feature Extraction
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import csr_matrix
        
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=100000)
        
        # Use target_sample for centroid
        target_texts = [d[:HEAD_CHARS] for d in target_sample]
        target_mats = vec.fit_transform(target_texts)
        
        # Variance calculation for "informative" docs
        # Since we need variance per doc, we look at the L2 norm of the tf-idf vector
        norms = np.array(target_mats.power(2).sum(axis=1)).ravel()
        # Rank by norm as a proxy for 'information' / magnitude in the vector space
        indices = np.argsort(norms)[::-1]
        top_count = max(1, len(indices) // 2)
        informative_indices = indices[:top_count]
        
        # Calculate centroid
        centroid = np.asarray(target_mats[informative_indices].mean(axis=0)).flatten()
        
        # Score pool docs by cosine similarity to centroid
        pool_texts = [d[:HEAD_CHARS] for d in docs]
        pool_mats = vec.transform(pool_texts)
        
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        # vec.transform is already normalized to some extent via sublinear_tf, 
        # but we calculate explicitly for accuracy.
        dot_products = (pool_mats.dot(centroid)).toarray().flatten()
        pool_norms = np.linalg.norm(pool_mats.data, ord=2) # Approximate via data
        # Better: calculate true norms
        pool_norms = np.sqrt(np.array(pool_mats.power(2).sum(axis=1)).ravel())
        centroid_norm = np.linalg.norm(centroid)
        
        scores = np.divide(dot_products, pool_norms * centroid_norm, 
                            out=np.zeros_like(dot_products), where=(pool_norms * centroid_norm) != 0)
        
        # Apply scores to survivors
        score_list = []
        for i in survivors:
            score_list.append(scores[i])
            
    except Exception:
        score_list = [0.0] * len(survivors)

    # Rank
    ranked = sorted(range(len(survivors)), key=lambda x: (-score_list[x], survivors[x]))
    ranked_indices = [survivors[i] for i in ranked]
    
    counter = _TokenCounter(docs, t0, deadline=150.0)
    top, _ = counter.take(ranked_indices, T)
    
    # Final selection logic
    if top:
        chosen = top
    else:
        chosen = ranked_indices
        
    rng.shuffle(chosen)
    return chosen
