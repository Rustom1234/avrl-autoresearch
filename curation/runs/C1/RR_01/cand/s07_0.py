"""Curator: Discriminative classifier ranking, near-dedup, and stratified ordering.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import time
import random
import hashlib
import re

from common import T

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

MAX_SCORE_TOKENS = 2000
MIN_TOKENS = 50
BUDGET_FACTOR = 1.02
TIME_LIMIT_ML = 120.0
BATCH = 512

def _get_clean_words(text):
    # Lowercased alphanumeric-only words
    return re.findall(r'[a-z0-9]+', text.lower())

def _get_fingerprint(text):
    words = _get_clean_words(text)
    # Try first 30 words
    f1 = " ".join(words[:30])
    # Try words 100-130
    f2 = " ".join(words[100:130])
    return hashlib.md5(f1.encode("utf-8", "ignore")).hexdigest(), \
           hashlib.md5(f2.encode("utf-8", "ignore")).hexdigest()

def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")

def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Initial Scoring
    vocab = set()
    bigrams = set()
    for doc in target_sample:
        toks = doc.split()
        vocab.update(toks)
        bigrams.update(zip(toks, toks[1:]))

    seen_exact = set()
    scored_indices = []
    scores = []
    
    # First pass: basic scoring and exact dedup
    for i, text in enumerate(docs):
        toks = text.split()
        if len(toks) < MIN_TOKENS:
            continue
        
        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen_exact:
            continue
        seen_exact.add(key)

        head = toks[:MAX_SCORE_TOKENS]
        n = len(head)
        u = sum(1 for t in head if t in vocab) / n
        b = (sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1) if n > 1 else 0.0)
        s = b + 0.25 * u + (0.1 if _has_title_line(text) else 0.0)
        
        scored_indices.append(i)
        scores.append((s, i))

    if not scored_indices:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Sort by score descending to find top/bottom percentiles
    scored_indices.sort(key=lambda x: (-scores[x][0], x))
    sorted_indices = [x for _, x in sorted(scores, key=lambda x: (-x[0], x[1]))]
    
    # Time check before ML
    use_ml = (time.time() - t_start) < TIME_LIMIT_ML
    
    if use_ml:
        try:
            pos_texts = []
            # Positives: Target sample chunks
            for doc in target_sample:
                toks = doc.split()
                for i in range(0, len(toks), 400):
                    pos_texts.append(" ".join(toks[i : i + 400]))
            
            # Positives: Top 15% of pool
            num_top = max(1, int(len(sorted_indices) * 0.15))
            for i in range(num_top):
                idx = sorted_indices[i]
                pos_texts.append(" ".join(docs[idx].split()[:MAX_SCORE_TOKENS]))
            
            # Negatives: Bottom 50% of pool
            num_bot = int(len(sorted_indices) * 0.5)
            neg_texts = []
            for i in range(len(sorted_indices) - num_bot, len(sorted_indices)):
                idx = sorted_indices[i]
                neg_texts.append(" ".join(docs[idx].split()[:MAX_SCORE_TOKENS]))
            
            vec = HashingVectorizer(analyzer='word', token_pattern=r'\S+', lowercase=False, 
                                    ngram_range=(1,2), n_features=2**20, norm='l2', alternate_sign=False)
            X = vec.fit_transform(pos_texts + neg_texts)
            y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
            
            lr = LogisticRegression(C=10, max_iter=200, solver='liblinear', class_weight='balanced')
            lr.fit(X, y)
            
            X_pool = vec.transform([" ".join(docs[i].split()[:MAX_SCORE_TOKENS]) for i in sorted_indices])
            margins = lr.decision_function(X_pool)
            
            # Final Rank: margin first, score as tie-break
            final_rank_keys = []
            for i in range(len(sorted_indices)):
                idx = sorted_indices[i]
                final_rank_keys.append((margins[i], scores[idx][0], idx))
            final_rank_keys.sort(key=lambda x: (-x[0], -x[1]))
            ranked = [x[2] for x in final_rank_keys]
        except:
            ranked = sorted_indices
    else:
        ranked = sorted_indices

    # Near-duplicate removal
    final_ranked = []
    fingerprints_seen = set()
    for idx in ranked:
        f1, f2 = _get_fingerprint(docs[idx])
        if f1 not in fingerprints_seen and f2 not in fingerprints_seen:
            final_ranked.append(idx)
            fingerprints_seen.add(f1)
            fingerprints_seen.add(f2)

    # Budget Selection
    budget = BUDGET_FACTOR * T
    selected = []
    total_tokens = 0.0
    
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except: pass

    for idx in final_ranked:
        doc = docs[idx]
        n_tok = 0
        if enc and (time.time() - t_start) < TIME_LIMIT_ML:
            try:
                n_tok = len(enc.encode_ordinary(doc))
            except: pass
        if n_tok == 0: n_tok = len(doc) // 4
        
        selected.append(idx)
        total_tokens += n_tok
        if total_tokens >= budget:
            break

    # Stratified Ordering: 30% lowest rank first, then 70% top rank
    # (Lowest rank means largest index in the 'final_ranked' list as it was sorted best-to-worst)
    split_pt = max(0, int(len(selected) * 0.7))
    # The 'top' 70% are the first 'split_pt' elements of selected
    top_70 = selected[:split_pt]
    bottom_30 = selected[split_pt:]
    
    rng.shuffle(top_70)
    rng.shuffle(bottom_30)
    
    # The prompt asks for "lowest-ranked 30% first, then top 70%"
    return bottom_30 + top_70
