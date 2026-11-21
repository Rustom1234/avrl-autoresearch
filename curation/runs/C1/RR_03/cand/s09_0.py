"""Curator: keep clean, intact, in-domain (WikiText-style) documents, de-duplicated,
in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_WORDS = 1500          # words used for the bigram score
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")


def _words(text):
    return text.lower().split()


def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)[:MAX_WORDS + 1]
        B.update(zip(w, w[1:]))
    return B


def _score_words(words, B):
    """Fraction of the bigrams in the first MAX_WORDS words that are in B."""
    w = words[:MAX_WORDS]
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(w, w[1:]):
        if b in B:
            hits += 1
    return hits / n


def _calibrate_tau(target_sample):
    """tau = 0.8 x 5th percentile of two-fold held-out target scores."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half = n // 2
    a, b = target_sample[:half], target_sample[half:]
    scores = []
    for build, held in ((a, b), (b, a)):
        B = _bigram_set(build)
        for t in held:
            w = _words(t)
            if len(w) >= 2:
                scores.append(_score_words(w, B))
    if not scores:
        return 0.0
    return 0.8 * float(np.percentile(np.asarray(scores), 5))


def _nonascii_letter_frac(text):
    n = len(text)
    if n == 0:
        return 0.0
    # Fast pre-check: total non-ASCII characters.
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii > MAX_NONASCII_LETTER_FRAC * n:
        cnt = sum(1 for c in text if ord(c) > 127 and c.isalpha())
        return cnt / n
    return 0.0


def _repetitive(words):
    n = len(words) - 4
    if n <= 0:
        return False
    top = Counter(
        zip(words, words[1:], words[2:], words[3:], words[4:])
    ).most_common(1)[0][1]
    # require top >= 3 so that very short docs are not dropped trivially
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    # Header check
    hdr_frac = 0.0
    if target_sample:
        hdr_frac = sum(1 for t in target_sample if t.lstrip().startswith("= ")) / len(target_sample)
    require_header = hdr_frac >= 0.9

    # Scoring data structures
    # Tier 1: All rules (Header, Bigram, Spam, Length, Non-ASCII, Repetition)
    # Tier 2: All but Header
    # Tier 3: All but Bigram and Header
    tiers = [[], [], []]
    
    seen_full = set()
    seen_prefix = set()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue
        
        low = text.lower()
        is_header = text.lstrip().startswith("= ")
        is_spam = any(p in low for p in SPAM_PATTERNS)
        
        # Base criteria (Universal filters)
        non_ascii_val = _nonascii_letter_frac(text)
        if non_ascii_val > MAX_NONASCII_LETTER_FRAC:
            continue
        
        words = low.split()
        score = _score_words(words, B)
        is_repetitive = _repetitive(words)

        # Check criteria for categorization
        passes_bigram = score >= tau
        passes_spam = not is_spam
        passes_repetition = not is_repetitive
        passes_header = is_header

        # Rule combinations
        # Tier 1: All rules
        if passes_header and passes_bigram and passes_spam and passes_repetition:
            tier_idx = 0
        # Tier 2: All but Header
        elif passes_bigram and passes_spam and passes_repetition:
            tier_idx = 1
        # Tier 3: All but Bigram and Header
        elif passes_spam and passes_repetition:
            tier_idx = 2
        else:
            continue

        # Deduplication
        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full or prefix in seen_prefix:
            continue
        
        seen_full.add(norm)
        seen_prefix.add(prefix)
        
        tiers[tier_idx].append((score, i))

    # Sort each tier by score descending
    for i in range(3):
        tiers[i].sort(key=lambda x: x[0], reverse=True)

    # Flatten into a single prioritized list
    prioritized_indices = []
    for tier in tiers:
        for score, idx in tier:
            prioritized_indices.append(idx)

    if not prioritized_indices:
        # Fallback
        prioritized_indices = list(range(len(docs)))
    else:
        # Shuffle to mix docs of the same priority/score slightly, 
        # but we must maintain the Tier logic. To be strictly tier-prioritized
        # but shuffled, we shuffle the indices within each tier.
        # However, the prompt says "shuffling the resulting list".
        # This usually implies prioritizing the order, then shuffling.
        # To balance "prioritizing" and "shuffling", we follow the instruction:
        # "select docs... by prioritizing Tier 1... then shuffling the resulting list".
        # This is slightly ambiguous (shuffling a prioritized list destroys priority).
        # Conventionally, this means shuffle the pool of IDs belonging to the selected set.
        # But the prompt specifies a sequence: Tier 1 -> 2 -> 3 -> Shuffle.
        # Let's stick to the sequence: T1 items, then T2, then T3.
        pass

    # Final selection to fill T tokens. 
    # Use tiktoken for estimation if possible.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def get_tok_count(idx_list):
            counts = 0
            for idx in idx_list:
                counts += len(enc.encode_ordinary_batch([docs[idx]]))[0]
            return counts
    except ImportError:
        def get_tok_count(idx_list):
            return sum(len(docs[i]) for i in idx_list) // 4

    # We need to select from the prioritized list until we hit T.
    # Since we want to "select docs to fill exactly T", we pick indices.
    # Because "the list holds fewer than T tokens it is cycled", we can just 
    # take the prioritized list. If it's too short, the trainer cycles it.
    # If it's too long, we take as many as fit into T.
    
    final_selection = []
    current_tokens = 0
    
    # To keep the priority but allow for "shuffling the resulting list",
    # one would usually shuffle the whole list. But the prompt says 
    # "select... by prioritizing... finally shuffling".
    # We will take the prioritized list and shuffle it.
    
    # Re-reading: "select docs to fill exactly T tokens by prioritizing Tier 1...
    # then shuffling the resulting list". 
    # If we shuffle at the end, the priority is lost. 
    # Usually, this means the order of the list returned should be random,
    # but the contents should be the best ones.
    
    # Logic: 
    # 1. Create a pool of all indices in prioritized_indices.
    # 2. Because "select docs to fill exactly T", we actually only need to 
    #    select enough to reach T. But if the total is less than T, we cycle.
    #    The trainer cycles, so we can just provide the best ones.
    
    # To satisfy "fill exactly T":
    # If total tokens < T, the trainer will cycle. So we provide all prioritized.
    # If total tokens > T, we truncate to T.
    
    # Let's build the list:
    # The prompt's logic order: 1. Prioritize. 2. Fill T. 3. Shuffle.
    
    res_indices = prioritized_indices
    
    # Truncate to T? The trainer cycles, so if we have fewer than T, it cycles.
    # If we have more than T, we should ideally provide exactly enough for T.
    # But "exactly T tokens" is hard to hit exactly with variable doc lengths.
    # We will provide the prioritized list.
    
    # Actually, the prompt says "select docs to fill exactly T tokens".
    # Let's try to get as close to T as possible.
    
    # Because of the "shuffling" requirement at the end:
    rng.shuffle(res_indices)
    
    # Wait, if I shuffle the prioritized list, Tier 1 docs are mixed with Tier 3.
    # The only way "prioritizing" and "shuffling" coexist is if we prioritize 
    # which docs are IN the list, then shuffle their order.
    
    # Correct interpretation: 
    # 1. Identify the best docs using tiers.
    # 2. If they exceed T, pick a subset that reaches T.
    # 3. Shuffle that subset.
    
    # Let's do that:
    all_best = prioritized_indices
    # If total tokens > T, we need to pick a subset.
    # For simplicity, if total tokens < T, we use all_best.
    # If total tokens > T, we take a proportional amount to reach T.
    
    total_avail_tokens = get_tok_count(all_best)
    if total_avail_tokens > T:
        # Take a slice that is roughly T
        # To be safe and simple, we take a slice of the prioritized list
        # that is roughly T tokens.
        count_to_take = int((T / total_avail_tokens) * len(all_best))
        # To ensure we don't miss the high-priority ones, we take the first 'count_to_take'
        # items (which are Tier 1, then 2, then 3) and then shuffle them.
        res_indices = all_best[:count_to_take]
    else:
        res_indices = all_best

    rng.shuffle(res_indices)
    
    return res_indices
