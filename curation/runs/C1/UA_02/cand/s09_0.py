"""Curator: Density-Based Clustering approach.
Clusters pool docs via TF-IDF, identifies target-rich clusters, 
and ranks by proximity to those centroids.
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
    if not target_sample:
        idx = list(range(n))
        random.shuffle(idx)
        return idx

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity

        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), 
                                    sublinear_tf=True, max_features=100000, dtype=np.float32)
        
        # Pre-filter pool slightly for speed
        valid_indices = []
        for i, d in enumerate(docs):
            if not _is_junk(d):
                valid_indices.append(i)
        
        if not valid_indices: return []
        
        filtered_docs = [docs[i] for i in valid_indices]
        # Fit on pool + target_sample
        all_texts = [d for d in target_sample] + filtered_docs
        X_all = char_vec.fit_transform(all_texts)
        
        X_target = X_all[:len(target_sample)]
        X_pool = X_all[len(target_sample):]
        
        # Clustering
        # Use a small number of clusters to capture high-density regions
        num_clusters = min(10, len(target_sample))
        kmeans = KMeans(n_clusters=num_clusters, n_init=3, random_state=seed).fit(X_pool)
        centroids = kmeans.cluster_centers_
        
        # For each centroid, find density of target documents
        # Target density = average similarity of targets to that centroid
        # We want clusters where targets are "tight" and "close"
        target_densities = []
        for c in centroids:
            sims = cosine_similarity(X_target, c.reshape(1, -1))[0]
            target_densities.append(np.mean(sims))
        
        # Pick the best cluster centroid (highest target mean similarity)
        best_cluster_idx = np.argmax(target_densities)
        best_centroid = centroids[best_cluster_idx]
        
        # Rank pool docs by proximity to this high-density centroid
        pool_sims = cosine_similarity(X_pool, best_centroid.reshape(1, -1)).flatten()
        
        # Final scoring: proximity to target-rich centroid + deduplication
        scores = []
        seen = set()
        for i, idx in enumerate(valid_indices):
            if _dedup_key(docs[idx]) in seen:
                continue
            seen.add(_dedup_key(docs[idx]))
            scores.append((pool_sims[i], idx))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        ranked = [x[1] for x in scores]
        
        # Fill budget
        counter = _TokenCounter(docs, t0, deadline=150.0)
        chosen, _ = counter.take(ranked, T)
        
        random.shuffle(chosen)
        return chosen

    except Exception:
        idx = list(range(n))
        random.shuffle(idx)
        return idx
