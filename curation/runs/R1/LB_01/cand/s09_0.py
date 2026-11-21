"""Curator: dedupe + junk filter + coherence-weighted ranking.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import time
from collections import Counter
import numpy as np
from common import T

MAX_WORDS = 2000
NEAR_DUP_CHARS = 300
MIN_CHARS = 200
MAX_NON_LETTER_RATIO = 0.4
HEADER_BONUS = 0.1

SPAM_TEMPLATES = (
    "discount code |",
    "online store |",
    "Click here to order",
    "free shipping on all",
)

_ASCII_LETTERS_RE = re.compile(r"[A-Za-z]+")

def _first_words(text):
    return text.split(None, MAX_WORDS)[:MAX_WORDS]

def _has_header(text):
    t = text.lstrip()
    if not t.startswith("= "):
        return False
    first_line = t.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =") and len(first_line) > 3

def _score(text, target_unigrams, target_bigrams, mean_coh, std_coh):
    words = _first_words(text)
    n = len(words)
    if n < 2:
        return 0.0
    uni_hits = sum(1 for w in words if w in target_unigrams)
    bi_hits = sum(1 for b in zip(words, words[1:]) if b in target_bigrams)
    
    # Base metrics
    unig_rate = uni_hits / n
    big_rate = bi_hits / (n - 1)
    
    # Coherence calculation
    coherence = big_rate / (unig_rate + 0.01)
    
    # Base score components
    # Score = bigram_hit / (unigram_hit + 0.05) + 0.5 * bigram_hit (+0.1 for a '= Title =' header)
    base_score = (big_rate / (unig_rate + 0.05)) + 0.5 * big_rate
    if _has_header(text):
        base_score += HEADER_BONUS
        
    # Penalize coherence below (mean - std)
    # If coherence < (mean - std), we apply a multiplier penalty
    penalty = 1.0
    if coherence < (mean_coh - std_coh):
        # Gradually reduce score as coherence drops further below the threshold
        penalty = max(0.1, 1.0 - (mean_coh - std_coh - coherence) * 0.5)
        
    return base_score * penalty

def _is_junk(text):
    stripped = text.strip()
    n = len(stripped)
    if n < MIN_CHARS:
        return True
    non_letters = len(_ASCII_LETTERS_RE.sub("", stripped))
    if non_letters / n > MAX_NON_LETTER_RATIO:
        return True
    for tpl in SPAM_TEMPLATES:
        if tpl in text:
            return True
    return False

def _calibrate_target(target_sample):
    uni_df = Counter()
    bi_df = Counter()
    coherences = []
    
    for text in target_sample:
        words = text.split()
        u = set(words)
        b = set(zip(words, words[1:]))
        uni_df.update(u)
        bi_df.update(b)
        
        # Calculate coherence for each sample to get distribution
        w = _first_words(text)
        nw = len(w)
        if nw >= 2:
            uh = sum(1 for x in w if x in uni_df)
            bh = sum(1 for y in zip(w, w[1:]) if y in bi_df)
            coherences.append((bh / (nw - 1)) / (uh / nw + 0.01))
            
    u_set = set(uni_df)
    b_set = set(bi_df)
    
    mean_coh = np.mean(coherences) if coherences else 0.0
    std_coh = np.std(coherences) if coherences else 0.0
    
    # For logic thresholds, use 2nd percentile of scores
    loo_scores = []
    for text in target_sample:
        words = _first_words(text)
        n = len(words)
        if n < 2: continue
        uh = sum(1 for w in words if uni_df[w] > 1)
        bh = sum(1 for bg in zip(words, words[1:]) if bi_df[bg] > 1)
        loo_scores.append((bh / (n - 1)) / (uh / n + 0.05) + 0.5 * (bh / (n-1)))
    
    s_thresh = np.percentile(loo_scores, 2) if loo_scores else 0.0
    return s_thresh, mean_coh, std_coh, u_set, b_set

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Dedupe & Junk
    seen_exact = set()
    seen_prefix = set()
    survivors = []
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text: continue
        norm = " ".join(text.split())
        h = hash(norm)
        if h in seen_exact: continue
        p = hash(norm[:NEAR_DUP_CHARS])
        if p in seen_prefix: continue
        seen_exact.add(h)
        seen_prefix.add(p)
        if _is_junk(text): continue
        survivors.append(i)

    # 2. Calibration
    s_thresh, m_coh, std_coh, t_uni, t_bi = _calibrate_target(target_sample)

    # 3. Scoring & In-Domain Identification
    scores = {}
    in_domain = []
    for i in survivors:
        text = docs[i]
        words = _first_words(text)
        n = len(words)
        if n < 2: continue
        
        uh = sum(1 for w in words if w in t_uni)
        bh = sum(1 for bg in zip(words, words[1:]) if bg in t_bi)
        
        score = _score(text, t_uni, t_bi, m_coh, std_coh)
        scores[i] = score
        
        # Keep logic: header + base score threshold
        if _has_header(text) and score >= s_thresh:
            in_domain.append(i)

    # 4. Token Counting
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def get_counts(indices):
            return [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in indices])]
    except:
        def get_counts(indices):
            return [max(1, len(docs[i]) // 4) for i in indices]

    in_counts = get_counts(in_domain)
    in_total = sum(in_counts)

    # 5. Selection
    selected = []
    if in_total >= T:
        ranked = sorted(in_domain, key=lambda i: (-scores[i], i))
        curr = 0
        for i in ranked:
            selected.append(i)
            curr += in_counts[in_domain.index(i)]
            if curr >= T: break
    elif in_total >= 0.4 * T:
        selected = in_domain
    else:
        selected = list(in_domain)
        needed = int(0.4 * T) - in_total
        sorted_surv = sorted(survivors, key=lambda i: (-scores[i], i))
        for i in sorted_surv:
            if i in in_domain: continue
            selected.append(i)
            needed -= (len(docs[i]) // 4)
            if needed <= 0: break
            
    if not selected:
        selected = list(range(len(docs)))

    rng.shuffle(selected)
    return selected
