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

    score_thresh, ratio_thresh, target_unigrams, target_bigrams = _calibrate_threshold(target_sample)

    scores = {}
    for i in survivors:
        scores[i] = _score(docs[i], target_unigrams, target_bigrams)

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

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except:
        pass

    def count_tokens(idx_list):
        if enc is None:
            return [max(1, len(docs[i]) // 4) for i in idx_list]
        try:
            return [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in idx_list])]
        except:
            return [max(1, len(docs[i]) // 4) for i in idx_list]

    in_domain_counts = count_tokens(in_domain)
    in_domain_total = sum(in_domain_counts)

    selected = []
    if in_domain_total >= T:
        ranked_in_domain = sorted(in_domain, key=lambda i: (-scores[i], i))
        current_total = 0
        
        # Diversity filter: penalize documents with Jaccard similarity > 0.3
        # Use the top 100 unique words for similarity
        def get_top_words(idx):
            ws = set(_first_words(docs[idx]))
            return set(list(ws)[:100])

        fingerprints = {}
        for i in ranked_in_domain:
            fingerprints[i] = get_top_words(i)

        for i in ranked_in_domain:
            # Check similarity against already selected
            is_redundant = False
            f_i = fingerprints[i]
            for s_idx in selected:
                f_s = fingerprints[s_idx]
                intersection = len(f_i.intersection(f_s))
                union = len(f_i.union(f_s))
                if union > 0 and (intersection / union) > 0.3:
                    is_redundant = True
                    break
            
            if not is_redundant:
                c = in_domain_counts[in_domain.index(i)]
                selected.append(i)
                current_total += c
                if current_total >= T:
                    break
    elif in_domain_total >= 0.4 * T:
        selected = in_domain
    else:
        selected = list(in_domain)
        remaining_needed = int(0.4 * T) - in_domain_total
        sorted_survivors = sorted(survivors, key=lambda i: (-scores[i], i))
        current_total = in_domain_total
        fingerprints = {i: get_top_words(i) for i in survivors}
        for i in sorted_survivors:
            if i in in_domain: continue
            
            is_redundant = False
            f_i = fingerprints[i]
            for s_idx in selected:
                f_s = fingerprints[s_idx]
                intersection = len(f_i.intersection(f_s))
                union = len(f_i.union(f_s))
                if union > 0 and (intersection / union) > 0.3:
                    is_redundant = True
                    break
            
            if not is_redundant:
                c = max(1, len(docs[i]) // 4)
                selected.append(i)
                current_total += c
                if current_total >= int(0.4 * T):
                    break

    if not selected:
        selected = list(range(len(docs)))

    rng.shuffle(selected)
    return selected
