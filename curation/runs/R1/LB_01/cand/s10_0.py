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

def _score(text, target_unigrams, target_bigrams):
    words = _first_words(text)
    n = len(words)
    if n < 2:
        return 0.0
    uni_hits = sum(1 for w in words if w in target_unigrams)
    bi_hits = sum(1 for b in zip(words, words[1:]) if b in target_bigrams)
    return _combine(bi_hits / (n - 1), uni_hits / n, _has_header(text))

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
        loo_scores.append(s)
        loo_ratios.append((bi_hits / (n - 1)) / (uni_hits / n + 0.00001))

    score_thresh = 0.0
    ratio_thresh = 0.0
    if loo_scores:
        score_thresh = float(np.percentile(np.asarray(loo_scores), 2))
    if loo_ratios:
        ratio_thresh = float(np.percentile(np.asarray(loo_ratios), 2))
        
    return score_thresh, ratio_thresh, set(uni_df), set(bi_df)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe & Junk Filter
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

    # 2. Thresholding
    score_thresh, ratio_thresh, target_unigrams, target_bigrams = _calibrate_threshold(target_sample)

    # 3. Scoring
    scores = {}
    for i in survivors:
        scores[i] = _score(docs[i], target_unigrams, target_bigrams)

    # 4. Identify In-Domain set
    in_domain = []
    for i in survivors:
        text = docs[i]
        score = scores[i]
        words = _first_words(text)
        n = len(words)
        if n < 2: continue
        uni_hits = sum(1 for w in words if w in target_unigrams)
        bi_hits = sum(1 for bg in zip(words, words[1:]) if bg in target_bigrams)
        current_ratio = (bi_hits / (n - 1)) / (uni_hits / n + 0.00001)
        
        if _has_header(text) and score >= score_thresh and current_ratio >= (0.5 * ratio_thresh):
            in_domain.append(i)

    # 5. Token Counting
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        pass

    def get_token_count(idx):
        if enc is None:
            return max(1, len(docs[idx]) // 4)
        try:
            return len(enc.encode_ordinary_batch([docs[idx]])[0])
        except:
            return max(1, len(docs[idx]) // 4)

    in_domain_counts = [get_token_count(i) for i in in_domain]
    in_domain_total = sum(in_domain_counts)
    
    # 6. Greedy Fill Selection Logic
    selected = []
    
    # Sort in-domain by score descending
    ranked_in_domain = sorted(in_domain, key=lambda i: (-scores[i], i))
    
    if in_domain_total >= T:
        current_total = 0
        for i in ranked_in_domain:
            selected.append(i)
            current_total += in_domain_counts[in_domain.index(i)]
            if current_total >= T:
                break
    elif in_domain_total < 0.5 * T:
        # Select all in-domain first
        selected = list(ranked_in_domain)
        # Supplement with survivors
        other_survivors = [i for i in survivors if i not in set(in_domain)]
        other_survivors.sort(key=lambda i: (-scores[i], i))
        
        current_total = in_domain_total
        for i in other_survivors:
            c = get_token_count(i)
            if current_total + c <= T * 1.1:
                selected.append(i)
                current_total += c
                if current_total >= T:
                    break
            else:
                # Try to fill as much as possible without overshooting 10%
                # Since we can't split documents, we take what fits the condition
                pass
    else:
        # Case: 0.5T <= in_domain_total < T
        # The prompt implies we just take all in-domain if it doesn't meet the 0.5T rule
        # but the specific logic says: keep all in-domain.
        selected = list(ranked_in_domain)

    if not selected:
        selected = list(range(len(docs)))

    rng.shuffle(selected)
    return selected
