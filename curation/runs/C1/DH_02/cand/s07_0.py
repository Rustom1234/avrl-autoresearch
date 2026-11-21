"""Curator: single-pass, priority-filled, fully mixed selection.

Pipeline
  1. Hard junk drop (spam phrases, HTML/markup, too short, non-ASCII, exact dups).
  2. Vocabulary-coverage filter against the target sample (drops letter-scrambled
     and foreign-language text).
  3. Word-order score (mean adjacent-bigram PMI over the top-400 target tokens);
     low scorers are flagged SHUF, not dropped.
  4. Tiers: A = clean + WikiText-formatted, B = other clean prose sorted by TF-IDF
     cosine similarity to the target centroid, C = SHUF, split with the same
     WikiText-format test into C_wiki (word-shuffled but WikiText-formatted) and
     C_other (the rest).
  5. Fill to 1.02 T gpt2 tokens: A, then C_wiki (seeded random order), then B
     (descending TF-IDF similarity), then C_other; a second copy of A only if the
     pool is still short of T. Never more than 2 copies of anything.
  6. One seeded uniform shuffle of the chosen set (no curriculum, no blocks).

select(docs, target_sample) -> ordered list of indices into docs.
The seed for any randomness is the environment variable CURATE_SEED.

DEBUG (module-level dict) is refreshed on every call to select() with the gpt2
token totals of A, C_wiki, B and C_other and the tokens actually chosen from each.
"""
import hashlib
import os
import random
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from common import T

SPAM = ('discount code', 'coupon', 'click here to order', 'free shipping',
        'best price guaranteed', 'limited offer', 'buy now')
MARKUP = ('<html', '<!DOCTYPE', '<div')

MIN_CHARS = 200
MAX_BRACKET_FRAC = 0.03
MAX_NONASCII_FRAC = 0.10

COV_TOKENS = 400        # alphabetic tokens used for vocabulary coverage
COV_FACTOR = 0.75       # threshold = COV_FACTOR * 5th percentile of target coverage
PMI_VOCAB = 400         # top-K target tokens for the word-order score
PMI_TOKENS = 600        # tokens scored per document
PMI_MIN_PAIRS = 30      # fewer in-vocabulary pairs -> never flagged
PMI_ALPHA = 0.5         # additive smoothing on bigram counts
PREFIX_CHARS = 12000    # prefix that is tokenised per pool document
TFIDF_CHARS = 5000      # prefix used for TF-IDF similarity
FILL = 1.02
CHUNK = 2000            # documents per tiktoken batch

# Filled in by select(); purely in-memory diagnostics.
DEBUG = {}


def _pmi_table(counts):
    p = counts.astype(np.float64) + PMI_ALPHA
    p /= p.sum()
    row = p.sum(axis=1, keepdims=True)
    col = p.sum(axis=0, keepdims=True)
    return np.log(p / (row * col)).ravel()


def _pair_codes(ids, k):
    if ids.size < 2:
        return np.empty(0, dtype=np.int64)
    a, b = ids[:-1], ids[1:]
    m = (a >= 0) & (b >= 0)
    return a[m] * k + b[m]


def _coverage(tokens, vocab):
    alpha = [t for t in tokens if t.isalpha()][:COV_TOKENS]
    if not alpha:
        return 0.0
    return sum(1 for t in alpha if t in vocab) / len(alpha)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # ---------------- target statistics ----------------
    tgt_tokens = [t.lower().split() for t in target_sample]
    tok_counts = Counter()
    for toks in tgt_tokens:
        tok_counts.update(toks)

    # (2) vocabulary seen at least twice in the target sample
    vocab = {w for w, c in tok_counts.items() if c >= 2}
    tgt_cov = [_coverage(toks, vocab) for toks in tgt_tokens]
    cov_thr = COV_FACTOR * float(np.percentile(tgt_cov, 5)) if tgt_cov else 0.0

    # (3) bigram PMI over the top-K target tokens
    top = [w for w, _ in tok_counts.most_common(PMI_VOCAB)]
    k = max(len(top), 1)
    tid = {w: i for i, w in enumerate(top)}
    tget = tid.get

    tgt_ids = [np.fromiter((tget(t, -1) for t in toks), dtype=np.int64, count=len(toks))
               for toks in tgt_tokens]
    tgt_codes = [_pair_codes(ids, k) for ids in tgt_ids]
    total_counts = np.zeros(k * k, dtype=np.int64)
    for c in tgt_codes:
        if c.size:
            total_counts += np.bincount(c, minlength=k * k)
    pmi = _pmi_table(total_counts.reshape(k, k))

    # Threshold. Target docs are scored leave-one-out (their own bigrams removed
    # from the table) so the scores are comparable to unseen pool docs.
    real_scores, shuf_scores = [], []
    for ids, codes in zip(tgt_ids, tgt_codes):
        own = np.bincount(codes, minlength=k * k) if codes.size else 0
        loo = _pmi_table((total_counts - own).reshape(k, k))
        c = _pair_codes(ids[:PMI_TOKENS], k)
        if c.size >= PMI_MIN_PAIRS:
            real_scores.append(float(loo[c].mean()))
        perm = ids.tolist()
        rng.shuffle(perm)
        c = _pair_codes(np.asarray(perm[:PMI_TOKENS], dtype=np.int64), k)
        if c.size >= PMI_MIN_PAIRS:
            shuf_scores.append(float(loo[c].mean()))
    if real_scores and shuf_scores:
        pmi_thr = 0.5 * (float(np.percentile(real_scores, 5)) +
                         float(np.percentile(shuf_scores, 95)))
    else:
        pmi_thr = float("-inf")

    # ---------------- single pass over the pool ----------------
    seen = set()
    tier_a, tier_b, tier_c_wiki, tier_c_other = [], [], [], []
    for i, d in enumerate(docs):
        n = len(d)
        # (1) hard junk drop
        if n < MIN_CHARS:
            continue
        if any(m in d for m in MARKUP):
            continue
        brackets = d.count('<') + d.count('>') + d.count('{') + d.count('}')
        if brackets > MAX_BRACKET_FRAC * n:
            continue
        if not d.isascii():
            non_ascii = n - len(d.encode('ascii', 'ignore'))
            if non_ascii > MAX_NONASCII_FRAC * n:
                continue
        low = d.lower()
        if any(p in low for p in SPAM):
            continue
        key = hashlib.md5(' '.join(low.split()).encode('utf-8', 'ignore')).digest()
        if key in seen:
            continue
        seen.add(key)

        # head tokens (lowercased, whitespace-split)
        if n > PREFIX_CHARS:
            toks = low[:PREFIX_CHARS].split()
            if len(toks) > 1:
                toks.pop()  # possibly truncated last token
        else:
            toks = low.split()

        # (2) vocabulary coverage
        if _coverage(toks, vocab) < cov_thr:
            continue

        # (3) word-order score
        head = toks[:PMI_TOKENS]
        ids = np.fromiter((tget(t, -1) for t in head), dtype=np.int64, count=len(head))
        codes = _pair_codes(ids, k)
        is_shuf = codes.size >= PMI_MIN_PAIRS and float(pmi[codes].mean()) < pmi_thr

        # (4) tiers -- the same WikiText-format test splits both clean and SHUF docs
        punct = toks.count(',') + toks.count('.')
        wikitext = (d.lstrip().startswith('= ')
                    or ' @-@ ' in d
                    or (len(toks) > 0 and punct * 100 >= 2 * len(toks)))
        if is_shuf:
            if wikitext:
                tier_c_wiki.append(i)
            else:
                tier_c_other.append(i)
        elif wikitext:
            tier_a.append(i)
        else:
            tier_b.append(i)

    # B: sort by TF-IDF cosine similarity to the target centroid (descending)
    if tier_b and target_sample:
        try:
            texts = [docs[i][:TFIDF_CHARS] for i in tier_b]
            texts += [t[:TFIDF_CHARS] for t in target_sample]
            vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2,
                                  max_features=100000, dtype=np.float32)
            x = vec.fit_transform(texts)
            nb = len(tier_b)
            centroid = np.asarray(x[nb:].mean(axis=0)).ravel()
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                sims = np.asarray(x[:nb] @ (centroid / norm)).ravel()
                order = np.argsort(-sims, kind='stable')
                tier_b = [tier_b[j] for j in order]
        except ValueError:
            pass  # degenerate vocabulary: keep pool order

    # A, C_wiki and C_other have no internal ranking; randomise (seeded) so a
    # budget cut is unbiased.
    rng.shuffle(tier_a)
    rng.shuffle(tier_c_wiki)
    rng.shuffle(tier_c_other)

    # ---------------- (5) priority fill, counted in gpt2 tokens ----------------
    enc = tiktoken.get_encoding("gpt2")
    threads = max(1, min(8, os.cpu_count() or 1))
    ntok = {}

    def count(indices):
        missing = [i for i in indices if i not in ntok]
        if missing:
            encoded = enc.encode_ordinary_batch([docs[i] for i in missing],
                                                num_threads=threads)
            for i, e in zip(missing, encoded):
                ntok[i] = len(e)
        return [ntok[i] for i in indices]

    def tier_total(indices):
        # Chunked so the encoded token lists never pile up in memory.
        s_total = 0
        for s in range(0, len(indices), CHUNK):
            s_total += sum(count(indices[s:s + CHUNK]))
        return s_total

    budget = FILL * T

    def take(cands, total):
        picked = []
        for s in range(0, len(cands), CHUNK):
            if total >= budget:
                break
            chunk = cands[s:s + CHUNK]
            for i, n in zip(chunk, count(chunk)):
                if total >= budget:
                    break
                picked.append(i)
                total += n
        return picked, total

    # New priority: A, C_wiki, B, C_other.
    fill_order = (('A', tier_a), ('C_wiki', tier_c_wiki),
                  ('B', tier_b), ('C_other', tier_c_other))

    total = 0
    chosen = []
    picked_by = {}
    chosen_tokens = {}
    for name, tier in fill_order:
        before = total
        picked, total = take(tier, total)
        picked_by[name] = picked
        chosen_tokens[name] = total - before
        chosen += picked

    # Only if every survivor together is still short of T: second copy of A.
    second_a_docs = 0
    second_a_tokens = 0
    if total < T and picked_by['A']:
        before = total
        extra, total = take(picked_by['A'], total)
        second_a_docs = len(extra)
        second_a_tokens = total - before
        chosen += extra

    # Diagnostics (tokenises the whole surviving pool; in-memory only).
    DEBUG.clear()
    DEBUG.update({
        'seed': seed,
        'T': T,
        'budget_tokens': budget,
        'fill_order': [name for name, _ in fill_order],
        'tier_docs': {name: len(tier) for name, tier in fill_order},
        'tier_tokens': {name: tier_total(tier) for name, tier in fill_order},
        'chosen_docs': {name: len(picked_by[name]) for name, _ in fill_order},
        'chosen_tokens': dict(chosen_tokens),
        'second_copy_A_docs': second_a_docs,
        'second_copy_A_tokens': second_a_tokens,
        'total_chosen_tokens': total,
        'fallback_raw_pool': not chosen,
    })

    if not chosen:  # nothing survived; fall back to the raw pool
        chosen = list(range(len(docs)))

    # ---------------- (6) one seeded uniform shuffle ----------------
    rng.shuffle(chosen)
    return chosen
