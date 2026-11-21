"""Curator: dedup -> word-bigram PU classifier (two rounds) -> shuffle guard
-> same-title variant deletion -> top-T cut -> explicit quality-weighted padding
when the core is shorter than T.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

All guard-passing docs with score >= 0 are candidates. Within each group of
candidates sharing the same '= ... =' title line, only the cleanest member is kept
(ties: higher classifier score); the others are deleted. The remaining docs are
taken in score order until T tokens. If they hold fewer than T tokens, the list is
padded explicitly instead of letting the trainer cycle: every core doc appears
once, and extra copies go to the cleanest docs first.

If the time limit is at risk before the deletion step finishes, the result is the
shuffled top-T core without deletion and without padding.
"""
import hashlib
import math
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from common import T

CHUNK = 1500          # characters per chunk
MAX_CHUNKS = 4        # chunks per pool doc used for scoring
MIN_TAIL = 300        # ignore trailing chunks shorter than this
N_NEG_DOCS = 3000     # pool docs sampled as negatives
GUARD_CHARS = 8000    # prefix of each doc used by the shuffle guard
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
PAD_DEADLINE = 165.0  # seconds: do not start the clean/deletion/padding stage after this
PAD_ABORT = 172.0     # seconds: abandon the clean/deletion stage if it runs past this
CLEAN_CHARS = 8000    # prefix of each candidate doc used for the cleanliness value
MIN_DF = 5            # a word is known if it occurs in at least this many candidate docs


def _dedup(docs):
    """Keep the first occurrence per (normalised 400-char prefix, length bucket)."""
    seen = set()
    kept = []
    for i, d in enumerate(docs):
        if not d or not d.strip():
            continue
        head = " ".join(d[:2000].lower().split())[:400]
        bucket = int(math.log2(len(d) + 1) * 4)
        key = (hashlib.md5(head.encode("utf-8", "ignore")).digest(), bucket)
        if key in seen:
            continue
        seen.add(key)
        kept.append(i)
    return kept


def _all_chunks(text):
    out = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    if len(out) > 1 and len(out[-1]) < MIN_TAIL:
        out.pop()
    return out


def _fit(X_pos, X_neg):
    X = sparse.vstack([X_pos, X_neg], format="csr")
    y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def _doc_scores(clf, X_pool, owner, n):
    s = clf.decision_function(X_pool)
    tot = np.bincount(owner, weights=s, minlength=n)
    cnt = np.bincount(owner, minlength=n)
    return tot / np.maximum(cnt, 1)


def _rank(values):
    """Average ranks (1-based), ties share the same rank."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]


def _title_key(text):
    """Normalised first line if it looks like '= ... =', else None."""
    stripped = text.lstrip()
    nl = stripped.find("\n")
    line = (stripped if nl < 0 else stripped[:nl]).strip()
    if len(line) < 3 or not line.startswith("=") or not line.endswith("="):
        return None
    if not line.strip("= \t"):
        return None
    return " ".join(line.lower().split())


def _fmt_median(vals):
    return "%.4f" % float(np.median(vals)) if len(vals) else "n/a"


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedup.
    kept = _dedup(docs)
    n = len(kept)
    if n == 0:
        return list(range(len(docs)))

    # 2. Features. norm=None so that sublinear tf sees raw counts; the
    #    TfidfTransformer does the l2 normalisation.
    hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False,
                           ngram_range=(1, 2), n_features=2 ** 20,
                           alternate_sign=False, norm=None, dtype=np.float32)

    # Pool chunks, vectorised one chunk position at a time so that the number
    # of chunks per doc can be reduced if the time limit is at risk.
    mats, owners = [], []
    last = 0.0
    for j in range(MAX_CHUNKS):
        if j > 0 and (time.time() - t0) + last > VEC_DEADLINE:
            break
        ts = time.time()
        texts, own = [], []
        lo = j * CHUNK
        for pos, i in enumerate(kept):
            c = docs[i][lo:lo + CHUNK]
            if j == 0 or len(c) >= MIN_TAIL:
                texts.append(c)
                own.append(pos)
        if not texts:
            break
        mats.append(hv.transform(texts))
        owners.append(np.asarray(own, dtype=np.int64))
        last = time.time() - ts
    owner = np.concatenate(owners)
    X_pool = sparse.vstack(mats, format="csr")
    del mats

    pos_chunks = []
    for d in target_sample:
        if d and d.strip():
            pos_chunks.extend(_all_chunks(d))
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool).tocsr()
    X_pos = tfidf.transform(X_pos).tocsr()

    # 3. Round 1: negatives are random pool docs.
    neg = rng.sample(range(n), min(N_NEG_DOCS, n))
    mask = np.zeros(n, dtype=bool)
    mask[neg] = True
    clf = _fit(X_pos, X_pool[mask[owner]])

    # 4. Score, then round 2 with negatives from the bottom half.
    score = _doc_scores(clf, X_pool, owner, n)
    bottom = np.argsort(score, kind="stable")[: max(1, n // 2)].tolist()
    neg2 = rng.sample(bottom, min(N_NEG_DOCS, len(bottom)))
    mask = np.zeros(n, dtype=bool)
    mask[neg2] = True
    clf = _fit(X_pos, X_pool[mask[owner]])
    score = _doc_scores(clf, X_pool, owner, n)
    del X_pool, X_pos

    # 5. Shuffle guard: adjacent-word-pair coverage by the target bigram set.
    #    The target reference is leave-one-out: a pair counts for a target doc
    #    only if it also occurs in another target doc.
    df = Counter()
    for d in target_sample:
        w = d.split()
        df.update(set(zip(w, w[1:])))
    bigrams = set(df)

    def frac(text):
        w = text[:GUARD_CHARS].split()
        m = len(w) - 1
        if m < 1:
            return 0.0
        return sum(1 for p in zip(w, w[1:]) if p in bigrams) / m

    loo = []
    for d in target_sample:
        w = d[:GUARD_CHARS].split()
        m = len(w) - 1
        if m >= 1:
            loo.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
    thr = 0.5 * float(np.median(loo)) if loo else 0.0

    # 6. All guard-passing candidates with score >= 0, in score order, with
    #    gpt2 token counts. The loop does NOT stop at T, so that the title
    #    groups see every sibling; the T cut is applied after the deletion.
    #    Only if the time limit is at risk (past PAD_DEADLINE, when the
    #    deletion stage will not run anyway) does it stop once T is reached.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    ranked = np.argsort(-score, kind="stable")
    cand = [int(p) for p in ranked if score[p] >= 0]

    selected = []      # candidate doc indices (into docs), in score order
    tok = []           # token count per candidate doc
    sel_score = []     # classifier score per candidate doc
    bg_cache = {}      # doc index -> shuffle-guard bigram fraction
    total = 0
    B = 512
    for s in range(0, len(cand), B):
        if total >= T and (time.time() - t0) > PAD_DEADLINE:
            break
        batch = []
        for p in cand[s:s + B]:
            f = frac(docs[kept[p]])
            if f >= thr:
                batch.append(p)
                bg_cache[kept[p]] = f
        if not batch:
            continue
        texts = [docs[kept[p]] for p in batch]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
        for p, ln in zip(batch, lens):
            ln = max(1, int(ln))
            selected.append(kept[p])
            tok.append(ln)
            sel_score.append(float(score[p]))
            total += ln

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            ln = max(1, len(docs[kept[int(p)]]) // 4)
            selected.append(kept[int(p)])
            tok.append(ln)
            sel_score.append(float(score[int(p)]))
            total += ln
            if total >= T:
                break

    cand_tokens = total
    n_cand = len(selected)

    def cut_to_T(positions):
        """Prefix of positions (score order) holding at least T tokens, if possible."""
        out, acc = [], 0
        for k in positions:
            if acc >= T:
                break
            out.append(k)
            acc += tok[k]
        return out, acc

    # The core as it was before this change: score order, taken until T tokens.
    old_core, old_core_tokens = cut_to_T(range(n_cand))
    old_core_set = set(old_core)

    def fallback(reason):
        """Current behaviour without deletion: shuffled top-T core, no padding."""
        print("curate: candidates = %d (tokens/T = %.4f), core_tokens/T = %.4f, "
              "core docs = %d, no deletion, no padding (%s)"
              % (n_cand, cand_tokens / float(T), old_core_tokens / float(T),
                 len(old_core), reason), flush=True)
        out = [selected[k] for k in old_core]
        rng.shuffle(out)
        return out

    # 7. No time left for the clean / deletion stage.
    if (time.time() - t0) > PAD_DEADLINE:
        return fallback("time limit")

    # 8. Cleanliness of each candidate doc on its first CLEAN_CHARS characters.
    target_vocab = set()
    for d in target_sample:
        target_vocab.update(d.split())

    word_df = Counter()
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return fallback("time limit")
        word_df.update({w for w in docs[i][:CLEAN_CHARS].split() if w.isalpha()})
    known = {w for w, c in word_df.items() if c >= MIN_DF}
    known |= target_vocab
    del word_df

    oov = np.ones(n_cand, dtype=np.float64)
    bg = np.zeros(n_cand, dtype=np.float64)
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return fallback("time limit")
        text = docs[i]
        words = [w for w in text[:CLEAN_CHARS].split() if w.isalpha()]
        if words:
            oov[k] = sum(1 for w in words if w not in known) / len(words)
        f = bg_cache.get(i)
        bg[k] = frac(text) if f is None else f

    clean = _rank(bg) - _rank(oov)

    # 9. Title groups: in each group with two or more members keep only the
    #    member with the highest clean value (ties: higher classifier score)
    #    and delete the others. Untitled docs and singletons are untouched.
    groups = {}
    for k, i in enumerate(selected):
        if (k & 1023) == 0 and (time.time() - t0) > PAD_ABORT:
            return fallback("time limit")
        key = _title_key(docs[i])
        if key is not None:
            groups.setdefault(key, []).append(k)

    deleted = np.zeros(n_cand, dtype=bool)
    winner = np.zeros(n_cand, dtype=bool)
    n_multi = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_multi += 1
        best = members[0]
        for k in members[1:]:
            if (clean[k], sel_score[k]) > (clean[best], sel_score[best]):
                best = k
        winner[best] = True
        for k in members:
            if k != best:
                deleted[k] = True

    if (time.time() - t0) > PAD_ABORT:
        return fallback("time limit")

    tok_arr = np.asarray(tok, dtype=np.int64)
    n_deleted = int(deleted.sum())
    deleted_tokens = int(tok_arr[deleted].sum())
    n_deleted_old_core = sum(1 for k in old_core if deleted[k])
    deleted_tokens_old_core = sum(tok[k] for k in old_core if deleted[k])

    # 10. Remaining docs are still in score order; take until T tokens.
    remaining = [k for k in range(n_cand) if not deleted[k]]
    remaining_tokens = cand_tokens - deleted_tokens
    core, core_tokens = cut_to_T(remaining)
    n_core = len(core)
    n_new = sum(1 for k in core if k not in old_core_set)

    print("curate: candidates = %d (tokens/T = %.4f), title groups with >1 member = %d"
          % (n_cand, cand_tokens / float(T), n_multi), flush=True)
    print("curate: deleted docs = %d (tokens/T = %.4f); of these in the old top-T "
          "core = %d (tokens/T = %.4f)"
          % (n_deleted, deleted_tokens / float(T),
             n_deleted_old_core, deleted_tokens_old_core / float(T)), flush=True)
    print("curate: median oov deleted = %s, kept siblings = %s; "
          "median bg deleted = %s, kept siblings = %s (n deleted = %d, n kept = %d)"
          % (_fmt_median(oov[deleted]), _fmt_median(oov[winner]),
             _fmt_median(bg[deleted]), _fmt_median(bg[winner]),
             n_deleted, int(winner.sum())), flush=True)
    print("curate: core_tokens/T before deletion = %.4f (docs = %d), "
          "after deletion = %.4f (docs = %d, available tokens/T = %.4f, "
          "docs newly admitted = %d)"
          % (old_core_tokens / float(T), len(old_core),
             core_tokens / float(T), n_core, remaining_tokens / float(T), n_new),
          flush=True)

    out = [selected[k] for k in core]

    # 11. Core fills the budget: seeded shuffle of the core.
    if core_tokens >= T or n_core == 0:
        print("curate: no padding", flush=True)
        rng.shuffle(out)
        return out

    # 12. Padding: every core doc once, then extra copies in descending clean
    #     order, pass after pass, until T tokens.
    core_clean = np.asarray([clean[k] for k in core], dtype=np.float64)
    order = [int(j) for j in np.argsort(-core_clean, kind="stable")]
    copies = np.ones(n_core, dtype=np.int64)
    total = core_tokens
    if order and sum(tok[core[j]] for j in order) > 0:
        while total < T:
            for j in order:
                out.append(selected[core[j]])
                copies[j] += 1
                total += tok[core[j]]
                if total >= T:
                    break

    rep = copies > 1
    core_oov = oov[np.asarray(core, dtype=np.int64)]
    print("curate: padded docs = %d, padded tokens/T = %.4f, max copies = %d"
          % (len(out), total / float(T), int(copies.max())), flush=True)
    print("curate: median oov repeated = %s (n=%d), not repeated = %s (n=%d)"
          % (_fmt_median(core_oov[rep]), int(rep.sum()),
             _fmt_median(core_oov[~rep]), int((~rep).sum())), flush=True)

    # 13. One seeded shuffle over the whole padded list.
    rng.shuffle(out)
    return out
