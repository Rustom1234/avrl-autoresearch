"""Curator: BM25-style scoring using TF-IDF weights derived from target_sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
import time
import numpy as np
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

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
    head = d[:1000]
    n = len(head)
    if n == 0: return True
    non_ascii = n - len(head.encode("ascii", "ignore"))
    if non_ascii / n > 0.20: return True
    alpha_space = len(_NON_ALPHA_SPACE.sub("", head))
    if alpha_space / n < 0.60: return True
    if len(_CODE_CHARS.findall(head)) / n > 0.015: return True
    words = head.lower().split()[:400]
    if len(words) >= 50 and len(set(words)) / len(words) < 0.30: return True
    return False

def _dedup_key(d):
    return hash(" ".join(d[:2500].lower().split())[:500])

class _TokenCounter:
    def __init__(self, docs, t0, deadline):
        self.docs = docs
        self.cache = {}
        self.t0 = t0
        self.deadline = deadline
        self.toks_seen = 0
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None

    def count(self, idxs):
        missing = [i for i in idxs if i not in self.cache]
        if missing:
            if self.enc is not None and (time.time() - self.t0) < self.deadline:
                try:
                    texts = [self.docs[i] for i in missing]
                    toks = self.enc.encode_ordinary_batch(texts, num_threads=1)
                    for i, d, t in zip(missing, texts, toks):
                        self.cache[i] = len(t)
                        self.toks_seen += len(t)
                    return [self.cache[i] for i in idxs]
                except: pass
            for i in missing:
                self.cache[i] = max(1, int(len(self.docs[i]) / 4.0))
        return [self.cache[i] for i in idxs]

    def take(self, seq, target, start_total=0):
        out, tot = [], start_total
        for i in seq:
            c = self.cache.get(i, max(1, int(len(self.docs[i]) / 4.0)))
            out.append(i)
            tot += c
            if tot >= target: return out, tot
        return out, tot

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1. Filter Junk and Dedup
    survivors = []
    seen_keys = set()
    for i, d in enumerate(docs):
        if _is_junk(d): continue
        k = _dedup_key(d)
        if k in seen_keys: continue
        seen_keys.add(k)
        survivors.append(i)
    
    if not survivors:
        indices = list(range(n))
        rng.shuffle(indices)
        return indices

    # 2. BM25-style scoring
    # Identify 'important' terms from target_sample
    target_words = []
    for d in target_sample:
        words = re.findall(r'\w{3,}', d.lower())
        target_words.extend(words)
    
    # Frequency in target
    target_counts = Counter(target_words)
    # Total count in pool for IDF
    pool_word_counts = Counter()
    for d in docs:
        words = re.findall(r'\w{3,}', d.lower())
        pool_word_counts.update(words)

    # Score = Sum_{w in target} count(w in doc) * log(N / (1 + count(w in pool)))
    # We use a simplified weight: target_freq / (pool_freq + 1)
    weights = {}
    important_words = set(target_counts.keys())
    for w in important_words:
        # Weight is higher if word appears in target but rarely in pool
        weights[w] = target_counts[w] * np.log1p(len(docs) / (1 + pool_word_counts[w]))

    scores = np.zeros(n)
    for i, d in enumerate(docs):
        words = re.findall(r'\w{3,}', d.lower())
        word_counts = Counter(words)
        doc_score = 0.0
        for w, count in word_counts.items():
            if w in weights:
                doc_score += count * weights[w]
        scores[i] = doc_score

    # 3. Rank and Select
    ranked_indices = sorted(survivors, key=lambda i: (-scores[i], i))
    counter = _TokenCounter(docs, t0, deadline=150.0)
    
    # Try to fill budget with top ranked
    top_selected, total_toks = counter.take(ranked_indices, T)
    
    # If we didn't reach budget, the trainer cycles the list, 
    # but we want to ensure high quality.
    # The list returned is cycled by the trainer, so we return the ranked list.
    
    # Final randomization of the indices to avoid ordering bias in training
    # but keep the top quality content at the front of the cycle.
    # Actually, the trainer cycles the return. To get the best results, 
    # we return the ranked list as is.
    
    # Re-shuffle just to ensure i.i.d. if the list is long enough to be fully consumed
    # but since the trainer cycles, we want the highest scores first.
    return top_selected
