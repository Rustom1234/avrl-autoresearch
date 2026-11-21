"""Curator: single-pass, priority-filled, fully mixed selection.

Pipeline
  1. Hard junk drop (spam phrases, HTML/markup, too short, non-ASCII, exact dups).
     The same pass builds a pool document-frequency table of word tokens.
  2. Vocabulary-coverage filter against the target sample (drops letter-scrambled
     and foreign-language text).
  3. Word-order score (mean adjacent-bigram PMI over the top-400 target tokens);
     low scorers are flagged SHUF, not dropped.
  4. Tiers: A = clean + WikiText-formatted, B = other clean prose sorted by TF-IDF
     cosine similarity to the target centroid, C = SHUF.
  5. Target-calibrated outlier check on case-sensitive character statistics
     (upper, digit, space, wlen), typo rate (rare) and repeated lines (rep).
     A outliers move to tier D, B outliers are dropped, C is unchanged.
  6. Fill to 1.02 T gpt2 tokens: A, then B, then D, then C; a second copy of A only
     if the pool is still short of T. Never more than 2 copies of anything.
  7. One seeded uniform shuffle of the chosen set (no curriculum, no blocks).

select(docs, target_sample) -> ordered list of indices into docs.
The seed for any randomness is the environment variable CURATE_SEED.
Diagnostics of the last call are kept in the module-level dict `debug`.
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

# outlier check
RARE_MIN_LEN = 4        # alphabetic tokens at least this long are typo candidates
RARE_TOKENS = 600       # candidates scored per document
RANGE_WIDEN = 0.25      # widen the target [min, max] by this fraction on each side
RARE_MAX_FACTOR = 1.5   # rare bound = max(target max * factor, target p99 + margin)
RARE_P99_MARGIN = 0.03
REP_MIN_LINE = 20       # lines at least this long count for the repetition score
REP_MIN_LINES = 6       # fewer such lines -> rep = 0
REP_MAX = 0.4           # fixed bound for rep

RANGE_STATS = ('upper', 'digit', 'space', 'wlen')
STATS = RANGE_STATS + ('rare', 'rep')

_UP = bytes(range(ord('A'), ord('Z') + 1))
_LOW = bytes(range(ord('a'), ord('z') + 1))
_DIG = b'0123456789'
_WS = b' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f'   # what str.isspace accepts in ASCII

debug = {}


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


def _head_tokens(low, n):
    """Lowercased whitespace tokens of the PREFIX_CHARS prefix."""
    if n > PREFIX_CHARS:
        toks = low[:PREFIX_CHARS].split()
        if len(toks) > 1:
            toks.pop()  # possibly truncated last token
        return toks
    return low.split()


def _rare_candidates(toks):
    return [t for t in toks if len(t) >= RARE_MIN_LEN and t.isalpha()]


def _rare(toks, df, max_df):
    cand = _rare_candidates(toks)[:RARE_TOKENS]
    if not cand:
        return 0.0
    return sum(1 for t in cand if df.get(t, 0) <= max_df) / len(cand)


def _char_stats(p):
    """Case-sensitive (upper, digit, space, wlen) of a text prefix."""
    n = len(p)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if p.isascii():
        b = p.encode('ascii')
        up = n - len(b.translate(None, _UP))
        letters = up + n - len(b.translate(None, _LOW))
        dg = n - len(b.translate(None, _DIG))
        sp = n - len(b.translate(None, _WS))
    else:
        up = letters = dg = sp = 0
        for c in p:
            if c.isalpha():
                letters += 1
                if c.isupper():
                    up += 1
            elif c.isdigit():
                dg += 1
            elif c.isspace():
                sp += 1
    ntoks = len(p.split())
    upper = up / letters if letters else 0.0
    wlen = (n - sp) / ntoks if ntoks else 0.0   # non-space chars / tokens
    return upper, dg / n, sp / n, wlen


def _rep(p):
    lines = [s for s in (ln.strip() for ln in p.split('\n')) if len(s) >= REP_MIN_LINE]
    if len(lines) < REP_MIN_LINES:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    debug.clear()

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

    # ---------------- pass 1: hard junk drop + pool document frequency ----------------
    seen = set()
    kept = []
    df = Counter()
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
        kept.append(i)
        # only typo candidates are ever looked up, so only they are counted
        df.update(set(_rare_candidates(_head_tokens(low, n))))

    # ---------------- outlier calibration on the target sample ----------------
    inf = float("inf")
    bounds = {s: (-inf, inf) for s in STATS}
    bounds['rep'] = (-inf, REP_MAX)
    if target_sample:
        tgt_stats = {s: [] for s in RANGE_STATS}
        tgt_rare = []
        for t in target_sample:
            vals = _char_stats(t[:PREFIX_CHARS])
            for s, v in zip(RANGE_STATS, vals):
                tgt_stats[s].append(v)
            # target docs are not in the pool: rare means pool df == 0
            tgt_rare.append(_rare(_head_tokens(t.lower(), len(t)), df, 0))
        for s in RANGE_STATS:
            lo, hi = min(tgt_stats[s]), max(tgt_stats[s])
            w = RANGE_WIDEN * (hi - lo)
            bounds[s] = (lo - w, hi + w)
        rare_hi = max(max(tgt_rare) * RARE_MAX_FACTOR,
                      float(np.percentile(tgt_rare, 99)) + RARE_P99_MARGIN)
        bounds['rare'] = (-inf, rare_hi)

    # ---------------- pass 2: coverage, word order, tiers, outliers ----------------
    tier_a, tier_b, tier_c, tier_d = [], [], [], []
    out_stat_tier = {t: {s: 0 for s in STATS} for t in 'ABC'}
    out_docs_tier = {t: 0 for t in 'ABC'}
    for i in kept:
        d = docs[i]
        n = len(d)
        low = d.lower()
        toks = _head_tokens(low, n)

        # (2) vocabulary coverage
        if _coverage(toks, vocab) < cov_thr:
            continue

        # (3) word-order score
        head = toks[:PMI_TOKENS]
        ids = np.fromiter((tget(t, -1) for t in head), dtype=np.int64, count=len(head))
        codes = _pair_codes(ids, k)
        is_shuf = codes.size >= PMI_MIN_PAIRS and float(pmi[codes].mean()) < pmi_thr

        # (4) tiers
        if is_shuf:
            tier = 'C'
        else:
            punct = toks.count(',') + toks.count('.')
            wikitext = (d.lstrip().startswith('= ')
                        or ' @-@ ' in d
                        or (len(toks) > 0 and punct * 100 >= 2 * len(toks)))
            tier = 'A' if wikitext else 'B'

        # (5) outlier check (case-sensitive prefix); pool docs count themselves in df
        p = d[:PREFIX_CHARS]
        vals = _char_stats(p) + (_rare(toks, df, 1), _rep(p))
        outlier = False
        for s, v in zip(STATS, vals):
            lo, hi = bounds[s]
            if v < lo or v > hi:
                outlier = True
                out_stat_tier[tier][s] += 1
        if outlier:
            out_docs_tier[tier] += 1

        if tier == 'C':
            tier_c.append(i)            # unchanged, counts are diagnostic only
        elif tier == 'A':
            (tier_d if outlier else tier_a).append(i)
        elif not outlier:
            tier_b.append(i)            # B outliers are dropped

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

    # A, C and D have no internal ranking; randomise so a budget cut is unbiased.
    rng.shuffle(tier_a)
    rng.shuffle(tier_c)
    rng.shuffle(tier_d)

    # ---------------- (6) priority fill, counted in gpt2 tokens ----------------
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
        tot = 0
        for s in range(0, len(indices), CHUNK):
            tot += sum(count(indices[s:s + CHUNK]))
        return tot

    # Every tier is tokenised in full so the debug totals are exact.
    tier_tokens = {'A': tier_total(tier_a), 'B': tier_total(tier_b),
                   'C': tier_total(tier_c), 'D': tier_total(tier_d)}

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

    total = 0
    chosen_tokens, chosen_docs = {}, {}
    chosen = []
    chosen_a = []
    for name, cands in (('A', tier_a), ('B', tier_b), ('D', tier_d), ('C', tier_c)):
        before = total
        picked, total = take(cands, total)
        chosen += picked
        chosen_tokens[name] = total - before
        chosen_docs[name] = len(picked)
        if name == 'A':
            chosen_a = picked

    # Only if every survivor together is still short of T: second copy of A.
    chosen_tokens['A2'] = 0
    chosen_docs['A2'] = 0
    if total < T and chosen_a:
        before = total
        extra, total = take(chosen_a, total)
        chosen += extra
        chosen_tokens['A2'] = total - before
        chosen_docs['A2'] = len(extra)

    fallback = not chosen
    if fallback:  # nothing survived; fall back to the raw pool
        chosen = list(range(len(docs)))

    # ---------------- debug record (in memory only) ----------------
    debug.update({
        'pool_docs': len(docs),
        'after_hard_drop': len(kept),
        'bounds': dict(bounds),
        'outliers_per_stat_tier': out_stat_tier,          # tier -> stat -> docs
        'outliers_per_stat': {s: sum(out_stat_tier[t][s] for t in 'ABC')
                              for s in STATS},
        'outliers_per_tier': out_docs_tier,               # A -> moved to D, B -> dropped
        'moved_a_to_d': out_docs_tier['A'],
        'dropped_b': out_docs_tier['B'],
        'tier_docs': {'A': len(tier_a), 'B': len(tier_b),
                      'C': len(tier_c), 'D': len(tier_d)},
        'tier_tokens': tier_tokens,
        'chosen_docs': chosen_docs,
        'chosen_tokens': chosen_tokens,
        'total_chosen_tokens': total,
        'budget_tokens': budget,
        'fallback_raw_pool': fallback,
    })

    # ---------------- (7) one seeded uniform shuffle ----------------
    rng.shuffle(chosen)
    return chosen
