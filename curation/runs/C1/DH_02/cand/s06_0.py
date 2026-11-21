"""Curator: single-pass, priority-filled, fully mixed selection.

Pipeline
  1. Hard junk drop (spam phrases, HTML/markup, too short, non-ASCII, exact dups).
  1b. Internal repetition drop (fewer than 60% distinct hashed pieces).
  1c. Near-duplicate drop (>= 50% of hashed pieces, minimum 3, already seen in a
      longer document).  Docs are processed longest first so the most complete
      copy is kept.
  2. Vocabulary-coverage filter against the target sample, on up to three windows
     (start / middle / end).  Any failing window drops the doc.  The bracket and
     non-ASCII fractions are also enforced per window.
  3. Word-order score (mean adjacent-bigram PMI over the top-400 target tokens) on
     the same windows; a doc with any low-scoring window is flagged SHUF, not
     dropped.
  4. Tiers: A = clean + WikiText-formatted, B = other clean prose sorted by TF-IDF
     cosine similarity to the target centroid, C = SHUF.
  5. Fill to 1.02 T gpt2 tokens: A, then B, then C; a second copy of A only if the
     pool is still short of T. Never more than 2 copies of anything.
  6. One seeded uniform shuffle of the chosen set (no curriculum, no blocks).

select(docs, target_sample) -> ordered list of indices into docs.
The seed for any randomness is the environment variable CURATE_SEED.
Diagnostics of the last call are left in the module-level dict `debug`.
"""
import hashlib
import os
import random
import time
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
PMI_TOKENS = 600        # tokens scored per window
PMI_MIN_PAIRS = 30      # fewer in-vocabulary pairs -> never flagged
PMI_ALPHA = 0.5         # additive smoothing on bigram counts
PREFIX_CHARS = 12000    # window size (chars) for the damage checks
TFIDF_CHARS = 5000      # prefix used for TF-IDF similarity
FILL = 1.02
CHUNK = 2000            # documents per tiktoken batch

PIECE_LONG_LINE = 300   # lines longer than this are split further on ' . '
PIECE_MIN_CHARS = 60    # minimum normalised piece length that is hashed
NEAR_DUP_FRAC = 0.5     # fraction of pieces already seen -> near duplicate
NEAR_DUP_MIN = 3        # minimum number of already-seen pieces
REPEAT_MIN_PIECES = 5   # internal-repetition rule needs at least this many pieces
REPEAT_DISTINCT = 0.6   # minimum distinct fraction of pieces

DEBUG_TOKEN_DEADLINE = 90.0  # seconds; full tier token totals are skipped after this

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


def _pieces(low):
    """Hashes of the normalised lines / sentences of a lowercased document."""
    out = []
    for line in low.splitlines():
        if len(line) < PIECE_MIN_CHARS:
            continue
        parts = line.split(' . ') if len(line) > PIECE_LONG_LINE else (line,)
        for p in parts:
            if len(p) < PIECE_MIN_CHARS:
                continue
            p = ' '.join(p.split())
            if len(p) >= PIECE_MIN_CHARS:
                out.append(hash(p))
    return out


def _windows(n):
    """Start / middle / end windows of PREFIX_CHARS; only distinct ones if short."""
    w = PREFIX_CHARS
    if n <= w:
        return [(0, n)]
    if n <= 2 * w:
        return [(0, w), (n - w, n)]
    m = (n - w) // 2
    return [(0, w), (m, m + w), (n - w, n)]


def select(docs, target_sample):
    t_start = time.monotonic()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    debug.clear()
    removed = {'near_dup': 0, 'internal_repeat': 0, 'window_brackets': 0,
               'window_nonascii': 0, 'window_coverage': 0}
    flagged_late_shuf = 0   # SHUF only because of a non-head window

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

    # ---------------- pass 1: hard junk drop + exact duplicates ----------------
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        n = len(d)
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
        survivors.append(i)
    del seen
    debug['after_hard_drop'] = len(survivors)

    # ------- pass 2: internal repetition + near duplicates (longest first) -------
    seen_pieces = set()
    kept = []
    for i in sorted(survivors, key=lambda j: (-len(docs[j]), j)):
        hashes = _pieces(docs[i].lower())
        if hashes:
            uniq = set(hashes)
            if (len(hashes) >= REPEAT_MIN_PIECES
                    and len(uniq) < REPEAT_DISTINCT * len(hashes)):
                removed['internal_repeat'] += 1
                continue
            hits = len(uniq & seen_pieces)
            if hits >= NEAR_DUP_MIN and hits >= NEAR_DUP_FRAC * len(uniq):
                removed['near_dup'] += 1
                continue
            seen_pieces |= uniq
        kept.append(i)
    del seen_pieces
    kept.sort()  # back to pool order
    debug['after_dup_rules'] = len(kept)

    # ---------------- pass 3: multi-window damage checks + tiers ----------------
    tier_a, tier_b, tier_c = [], [], []
    for i in kept:
        d = docs[i]
        n = len(d)

        # per-window bracket / non-ASCII fractions (single window == whole doc,
        # which was already checked in pass 1)
        wins = _windows(n)
        if len(wins) > 1:
            bad = None
            for s, e in wins:
                seg = d[s:e]
                m = len(seg)
                br = seg.count('<') + seg.count('>') + seg.count('{') + seg.count('}')
                if br > MAX_BRACKET_FRAC * m:
                    bad = 'window_brackets'
                    break
                if not seg.isascii():
                    non_ascii = m - len(seg.encode('ascii', 'ignore'))
                    if non_ascii > MAX_NONASCII_FRAC * m:
                        bad = 'window_nonascii'
                        break
            if bad:
                removed[bad] += 1
                continue

        low = d.lower()
        nl = len(low)
        win_toks = []
        for s, e in _windows(nl):
            toks = low[s:e].split()
            if e < nl and len(toks) > 1:
                toks.pop()          # possibly truncated last token
            if s > 0 and len(toks) > 1:
                toks = toks[1:]     # possibly truncated first token
            win_toks.append(toks)

        # (2) vocabulary coverage: every window must pass
        if any(_coverage(toks, vocab) < cov_thr for toks in win_toks):
            removed['window_coverage'] += 1
            continue

        # (3) word-order score: any low window flags SHUF
        is_shuf = False
        for w, toks in enumerate(win_toks):
            head = toks[:PMI_TOKENS]
            ids = np.fromiter((tget(t, -1) for t in head), dtype=np.int64,
                              count=len(head))
            codes = _pair_codes(ids, k)
            if codes.size >= PMI_MIN_PAIRS and float(pmi[codes].mean()) < pmi_thr:
                is_shuf = True
                if w > 0:
                    flagged_late_shuf += 1
                break

        # (4) tiers
        if is_shuf:
            tier_c.append(i)
            continue
        toks = win_toks[0]
        punct = toks.count(',') + toks.count('.')
        wikitext = (d.lstrip().startswith('= ')
                    or ' @-@ ' in d
                    or (len(toks) > 0 and punct * 100 >= 2 * len(toks)))
        if wikitext:
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

    # A and C have no internal ranking; randomise so a budget cut is unbiased.
    rng.shuffle(tier_a)
    rng.shuffle(tier_c)

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
    chosen_a, total = take(tier_a, total)
    chosen_b, total = take(tier_b, total)
    chosen_c, total = take(tier_c, total)
    chosen = chosen_a + chosen_b + chosen_c

    # Only if every survivor together is still short of T: second copy of A.
    extra = []
    if total < T and chosen_a:
        extra, total = take(chosen_a, total)
        chosen += extra

    # ---------------- (4) diagnostics ----------------
    debug['removed'] = dict(removed)
    debug['shuf_flagged'] = len(tier_c)
    debug['shuf_flagged_non_head_window'] = flagged_late_shuf
    debug['tier_docs'] = {'A': len(tier_a), 'B': len(tier_b), 'C': len(tier_c)}
    debug['chosen_docs'] = {'A': len(chosen_a), 'B': len(chosen_b),
                            'C': len(chosen_c), 'A_second_copy': len(extra)}
    debug['chosen_tokens'] = {
        'A': sum(ntok[i] for i in chosen_a),
        'B': sum(ntok[i] for i in chosen_b),
        'C': sum(ntok[i] for i in chosen_c),
        'A_second_copy': sum(ntok[i] for i in extra),
    }
    debug['total_tokens'] = total
    tier_tokens = {}
    for name, tier in (('A', tier_a), ('B', tier_b), ('C', tier_c)):
        # Full totals need the unchosen remainder tokenised; skip if short on time.
        if time.monotonic() - t_start < DEBUG_TOKEN_DEADLINE:
            tier_tokens[name] = sum(count(tier))
        else:
            tier_tokens[name] = None
    debug['tier_tokens'] = tier_tokens

    if not chosen:  # nothing survived; fall back to the raw pool
        chosen = list(range(len(docs)))

    # ---------------- (6) one seeded uniform shuffle ----------------
    rng.shuffle(chosen)
    debug['seconds'] = time.monotonic() - t_start
    return chosen
