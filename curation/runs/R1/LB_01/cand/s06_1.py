import os
import random
import re
from collections import Counter

import numpy as np

from common import T

MAX_WORDS = 2000
NEAR_DUP_CHARS = 300
MIN_CHARS = 200
MAX_NON_LETTER_RATIO = 0.4
HEADER_BONUS = 0.1
TIME_BUDGET_FOR_TOKENISING = 120.0

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

    if not survivors:
        return list(range(len(docs)))

    # 2. Scoring
    score_thresh, ratio_thresh, target_unigrams, target_bigrams = _calibrate_threshold(target_sample)
    
    scores = {}
    for i in survivors:
        scores[i] = _score(docs[i], target_unigrams, target_bigrams)

    # Sort survivors by score descending
    sorted_survivors = sorted(survivors, key=lambda i: (-scores[i], i))
    
    # Use tiktoken if available for counting
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        pass

    def get_count(idx):
        if enc:
            return len(enc.encode_ordinary_batch([docs[idx]])[0])
        return max(1, len(docs[idx]) // 4)

    # Selection Logic
    selected_indices = []
    current_tokens = 0
    
    # Calculate totals for top 10%
    top_n_count = max(1, int(len(sorted_survivors) * 0.1))
    top_n_indices = sorted_survivors[:top_n_count]
    
    for idx in top_n_indices:
        selected_indices.append(idx)
        current_tokens += get_count(idx)

    remaining_sorted = [i for i in sorted_survivors if i not in top_n_indices]
    
    def fill_to_target(target_val):
        nonlocal current_tokens
        i = 0
        while i < len(remaining_sorted) and current_tokens < target_val:
            idx = remaining_sorted[i]
            selected_indices.append(idx)
            current_tokens += get_count(idx)
            i += 1

    # Step 2: Ensure 0.5T
    fill_to_target(0.5 * T)
    
    # Step 3: Fill to T
    fill_to_target(T)

    # Final Shuffle
    rng.shuffle(selected_indices)
    return selected_indices
