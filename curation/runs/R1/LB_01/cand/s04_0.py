"""Curator: dedupe + junk filter + diversity-sensitive ranking against the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact duplicates and near-duplicates.
  2. Hard junk filters.
  3. Score and measure bigram/unigram density against target_sample.
  4. Filter for 'in-domain' docs using a high-quality diversity filter:
     - Must have '= Title =' header.
     - Score >= 2nd percentile of leave-one-out scores.
     - bigram_hit / unigram_hit must be within a high-confidence range of the target_sample.
  5. Select subset based on token counts.
  6. Shuffle and return.
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

def _combine(bigram_hit, unigram_hit, header):
    score = bigram_hit / (unigram_hit + 0.05) + 0.5 * bigram_hit
    if header:
        score += HEADER_BONUS
    return score

def _score_data(words, target_unigrams, target_bigrams, header_flag):
    n = len(words)
    if n < 2:
        return 0.0, 0.0, 0.0
    uni_hits = sum(1 for w in words if w in target_unigrams)
    bi_hits = sum(1 for b in zip(words, words[1:]) if b in target_bigrams)
    
    score = _combine(bi_hits / (n - 1), uni_hits / n, header_flag)
    ratio = (bi_hits / (n - 1)) / (uni_hits / n + 0.00001)
    return score, ratio, (uni_hits / n)

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

def _calibrate_threshold(target_sample):
    uni_df = Counter()
    bi_df = Counter()
    for text in target_sample:
        words = text.split()
        u = set(words)
        b = set(zip(words, words[1:]))
        uni_df.update(u)
        bi_df.update(b)

    loo_scores = []
    loo_ratios = []
    for text in target_sample:
        words = _first_words(text)
        n = len(words)
        if n < 2:
            continue
        uni_hits = sum(1 for w in words if uni_df[w] > 1)
        bi_hits = sum(1 for bg in zip(words, words[1:]) if bi_df[bg] > 1)
        
        s = _combine(bi_hits / (n - 1), uni_hits / n, _has_header(text))
        r = (bi_hits / (n - 1)) / (uni_hits / n + 0.00001)
        loo_scores.append(s)
        loo_ratios.append(r)

    score_thresh = 0.0
    ratio_mean = 0.0
    ratio_std = 0.0
    
    if loo_scores:
        score_thresh = float(np.percentile(np.array(loo_scores), 2))
    if loo_ratios:
        r_arr = np.array(loo_ratios)
        ratio_mean = np.mean(r_arr)
        ratio_std = np.std(r_arr)
        
    return score_thresh, ratio_mean, ratio_std, set(uni_df), set(bi_df)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe & Junk
    seen_exact = set()
    seen_prefix = set()
    survivors = []
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        norm = " ".join(text.split())
        h = hash(norm)
        if h in seen_exact:
            continue
        p = hash(norm[:NEAR_DUP_CHARS])
        if p in seen_prefix:
            continue
        seen_exact.add(h)
        seen_prefix.add(p)
        if _is_junk(text):
            continue
        survivors.append(i)

    # 2. Calibration
    score_thresh, r_mean, r_std, target_unigrams, target_bigrams = _calibrate_threshold(target_sample)

    # 3. Scoring & Filtering
    # Diversity filter: reject if ratio < (mean - 2 * std)
    diversity_lower_bound = r_mean - 2 * r_std
    
    scores = {}
    in_domain = []
    
    for i in survivors:
        text = docs[i]
        words = _first_words(text)
        header_flag = _has_header(text)
        score, ratio, _ = _score_data(words, target_unigrams, target_bigrams, header_flag)
        scores[i] = score
        
        if header_flag and score >= score_thresh and ratio >= diversity_lower_bound:
            in_domain.append(i)

    # 4. Token Counting
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    def get_c(idx):
        if enc:
            try:
                return len(enc.encode_ordinary_batch([docs[idx]])[0])
            except:
                return len(docs[idx]) // 4
        return len(docs[idx]) // 4

    in_domain_counts = [get_c(i) for i in in_domain]
    in_domain_total = sum(in_domain_counts)

    # 5. Selection
    selected = []
    if in_domain_total >= T:
        ranked = sorted(in_domain, key=lambda i: (-scores[i], i))
        curr = 0
        for i in ranked:
            selected.append(i)
            curr += in_domain_counts[in_domain.index(i)]
            if curr >= T:
                break
    elif in_domain_total >= 0.4 * T:
        selected = in_domain
    else:
        selected = list(in_domain)
        sorted_surv = sorted(survivors, key=lambda i: (-scores[i], i))
        curr = in_domain_total
        target_min = int(0.4 * T)
        for i in sorted_surv:
            if i in in_domain: continue
            selected.append(i)
            curr += get_c(i)
            if curr >= target_min:
                break

    if not selected:
        selected = list(range(len(docs)))

    rng.shuffle(selected)
    return selected
