"""Curator: filtered, target-matched selection.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized hash) + near dedupe (first 200 chars)
  2. junk removal (too short, repeated lines, highly compressible)
  3. whole-document damage detection on up to 4 windows of 300 words
     (start, 1/3, 2/3, end): word-bigram hit rate against the target sample,
     function-word/punctuation adjacency rate, share of ordinary characters,
     mean word length. A doc is dropped if any window fails.
  4. domain classifier (char n-gram TF-IDF + logistic regression) on a snippet
     built from the same windows
  5. keep prob > 0.5; count tokens (tiktoken gpt2); if the kept set is shorter
     than T tokens, return several independently seeded shuffles of it
     (random.Random(seed + k) for epoch k). No off-domain padding.
"""
import os
import random
import time
import zlib
import hashlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MIN_CHARS = 200
SCORE_CHAR_BUDGET = 8_000_000   # total chars pushed through the char n-gram vectorizer
MAX_POS = 3000
MAX_NEG = 3000
N_BIGRAMS = 300

WIN_WORDS = 300                 # words per inspection window
MAX_WINDOWS = 4
MIN_BIGRAMS = 30                # below this, bigram statistics are too noisy
MIN_OK_CHAR_FRAC = 0.85
MIN_MEAN_WORD = 3.0
MAX_MEAN_WORD = 8.0

SCORE_DEADLINE = 125            # seconds; classifier scoring stops here
TOKEN_DEADLINE = 160            # seconds; token counting falls back to an estimate
MAX_EPOCHS = 10_000
MAX_OUTPUT_LEN = 5_000_000

_PUNCT = {",", ".", ";"}
_FUNC = {
    "the", "of", "a", "an", "in", "to", "and", "by", "with", "for", "from",
    "at", "on", "as", "its", "their", "his", "her",
}
_PP = {(",", "."), (".", ","), (",", ",")}

_OK_BYTES = (
    b"abcdefghijklmnopqrstuvwxyz"
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    b"0123456789"
    b" \t\n"
    b".,;:!?'\"()[]{}-/&%$#@*+=<>_"
)


def _has_marker(d):
    """WikiText-style formatting: '= Title =' first line plus spaced punctuation."""
    s = d.lstrip()
    if not s.startswith("= "):
        return False
    head = d[:20000]
    return (" @-@ " in head) or (" , " in head) or (" . " in head)


def _is_junk(d):
    if len(d) < MIN_CHARS:
        return True
    lines = [ln.strip() for ln in d.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.5:
        return True
    raw = d[:50000].encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw, 6)) / len(raw) < 0.2:
        return True
    return False


def _first_bigrams(d, n=N_BIGRAMS):
    w = d[:n * 40].lower().split()[: n + 1]
    return list(zip(w, w[1:]))


def _adjacency_rate(bgs):
    if not bgs:
        return 0.0
    c = 0
    for a, b in bgs:
        if (a in _FUNC and b in _PUNCT) or ((a, b) in _PP):
            c += 1
    return c / len(bgs)


def _windows(words):
    """Up to 4 windows of WIN_WORDS words: start, 1/3, 2/3, end.

    Short docs get fewer windows (ceil(n / WIN_WORDS)), always including the
    start and, when there is more than one window, the end.
    """
    n = len(words)
    if n <= WIN_WORDS:
        return [words]
    k = min(MAX_WINDOWS, -(-n // WIN_WORDS))
    span = n - WIN_WORDS
    out = []
    for j in range(k):
        s = (j * span) // (k - 1)
        out.append(words[s:s + WIN_WORDS])
    return out


def _window_ok(w, doc_count, hit_thr, adj_thr):
    nw = len(w)
    if nw == 0:
        return False

    # character-level noise: mean word length
    mean_len = sum(map(len, w)) / nw
    if mean_len < MIN_MEAN_WORD or mean_len > MAX_MEAN_WORD:
        return False

    # character-level noise: share of ordinary characters
    s = " ".join(w)
    b = s.encode("ascii", "ignore")
    good = len(b) - len(b.translate(None, _OK_BYTES))
    if good < MIN_OK_CHAR_FRAC * len(s):
        return False

    # word-level damage: shuffles, spliced spam
    if hit_thr is not None and nw - 1 >= MIN_BIGRAMS:
        bgs = list(zip(w, w[1:]))
        hit = sum(1 for bg in bgs if bg in doc_count) / len(bgs)
        if hit < hit_thr:
            return False
        if _adjacency_rate(bgs) > adj_thr:
            return False
    return True


def _doc_ok(d, doc_count, hit_thr, adj_thr):
    words = d.lower().split()
    if not words:
        return False
    for w in _windows(words):
        if not _window_ok(w, doc_count, hit_thr, adj_thr):
            return False
    return True


def _snippet(d, L):
    """Classifier text built from the same windows used for damage checks.

    With 4 windows each contributes L/4 chars; docs with fewer windows split
    the same L-char budget across the windows they have. Text is lowercased
    and whitespace-normalized, which the char_wb vectorizer would do anyway.
    """
    wins = _windows(d.lower().split())
    per = max(1, L // len(wins))
    return " ".join(" ".join(w)[:per] for w in wins)


def _count_tokens(docs, idx, t0):
    """Return (token_count, exact). Stops early once T tokens are reached."""
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    total = 0.0
    chars = 0
    exact = True
    B = 256
    for s in range(0, len(idx), B):
        if enc is None or time.time() - t0 > TOKEN_DEADLINE:
            ratio = (total / chars) if chars else 0.25
            total += ratio * sum(len(docs[i]) for i in idx[s:])
            exact = False
            break
        batch = [docs[i] for i in idx[s:s + B]]
        try:
            for toks in enc.encode_ordinary_batch(batch):
                total += len(toks)
        except Exception:
            enc = None
            ratio = (total / chars) if chars else 0.25
            total += ratio * sum(len(docs[i]) for i in idx[s:])
            exact = False
            break
        chars += sum(len(x) for x in batch)
        if total >= T:
            break
    return total, exact


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- 1 + 2: dedupe and junk filtering --------------------------------
    seen_exact = set()
    seen_head = set()
    cand = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        he = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if he in seen_exact:
            continue
        seen_exact.add(he)
        hh = hashlib.md5(norm[:200].lower().encode("utf-8", "ignore")).digest()
        if hh in seen_head:
            continue
        seen_head.add(hh)
        if _is_junk(d):
            continue
        cand.append(i)
    del seen_exact, seen_head

    if not cand:
        return _fallback(range(n))

    # ---- 3: whole-document damage detection ------------------------------
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    doc_count = {}
    hit_thr = None
    adj_thr = None
    if len(targets) >= 5:
        for t in targets:
            w = t.lower().split()
            for bg in set(zip(w, w[1:])):
                doc_count[bg] = doc_count.get(bg, 0) + 1

        loo, tadj = [], []
        for t in targets:
            bgs = _first_bigrams(t)
            if len(bgs) < MIN_BIGRAMS:
                continue
            # leave-one-out: bigram must occur in at least one *other* target doc
            loo.append(sum(1 for bg in bgs if doc_count.get(bg, 0) >= 2) / len(bgs))
            tadj.append(_adjacency_rate(bgs))

        if len(loo) >= 5:
            hit_thr = float(np.percentile(loo, 5))
            adj_thr = max(0.02, 3.0 * float(np.percentile(tadj, 99)))

    kept = [i for i in cand if _doc_ok(docs[i], doc_count, hit_thr, adj_thr)]
    survivors = kept if kept else cand
    del doc_count

    # ---- 4: domain classifier --------------------------------------------
    final = survivors
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Positives come from docs that passed the whole-document checks, so
        # damaged text is never taught as on-domain; negatives from all
        # candidates without the formatting marker.
        pos_pool = [i for i in survivors if _has_marker(docs[i])]
        neg_pool = [i for i in cand if not _has_marker(docs[i])]
        crng = random.Random(seed + 1)
        if len(pos_pool) > MAX_POS:
            pos_pool = crng.sample(pos_pool, MAX_POS)
        n_neg = min(len(neg_pool), MAX_NEG)
        neg_s = crng.sample(neg_pool, n_neg) if n_neg else []

        if targets and len(neg_s) >= 10:
            n_train = len(targets) + len(pos_pool) + len(neg_s)
            L = int(SCORE_CHAR_BUDGET / max(1, n_train + len(survivors)))
            L = max(300, min(1500, L))

            train_txt = (
                [_snippet(t, L) for t in targets]
                + [_snippet(docs[i], L) for i in pos_pool]
                + [_snippet(docs[i], L) for i in neg_s]
            )
            y = np.array([1] * (len(targets) + len(pos_pool)) + [0] * len(neg_s))

            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                sublinear_tf=True,
                max_features=200_000,
                dtype=np.float32,
            )
            Xtr = vec.fit_transform(train_txt)
            clf = LogisticRegression(
                C=10.0, max_iter=200, class_weight="balanced", random_state=seed
            )
            clf.fit(Xtr, y)
            del Xtr, train_txt

            probs = np.zeros(len(survivors), dtype=np.float64)
            B = 2000
            done = True
            s = 0
            for s in range(0, len(survivors), B):
                if time.time() - t0 > SCORE_DEADLINE:  # leave room for token counting
                    done = False
                    break
                chunk = [_snippet(docs[i], L) for i in survivors[s:s + B]]
                probs[s:s + B] = clf.predict_proba(vec.transform(chunk))[:, 1]

            if done:
                picked = [i for i, p in zip(survivors, probs) if p > 0.5]
            else:
                # out of time: keep scored-positive docs, and for unscored docs
                # fall back to the formatting marker
                picked = []
                for k, i in enumerate(survivors):
                    if k < s:
                        if probs[k] > 0.5:
                            picked.append(i)
                    elif _has_marker(docs[i]):
                        picked.append(i)
            if picked:
                final = picked
    except Exception:
        final = survivors

    if not final:
        final = survivors if survivors else cand

    # ---- 5: fill the T budget with independently shuffled epochs ----------
    final = list(final)
    total, exact = _count_tokens(docs, final, t0)

    if total >= T:
        n_epochs = 1
    elif total <= 0:
        n_epochs = 1
    else:
        n_epochs = int(-(-T // total))
        if not exact:
            n_epochs += 1  # margin for an estimated count; extra is never read
    n_epochs = max(1, min(n_epochs, MAX_EPOCHS, max(1, MAX_OUTPUT_LEN // len(final))))

    out = []
    for k in range(n_epochs):
        order = list(final)
        random.Random(seed + k).shuffle(order)
        out.extend(order)
    return out
