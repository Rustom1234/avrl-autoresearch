"""Wikitext-first curator: deduplicated, damage-filtered, topped up by score if needed.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import hashlib
import os
import random
import re
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WORD_RE = re.compile(r'\w+')
_WIKI_MARKERS = (' , ', ' . ', ' @-@ ', ' ( ')
_PREFIX_TOKENS = 1500
_MIN_WIKI_TOKENS = 100
_MAD_FLOOR = 1e-3  # avoids a degenerate threshold when MAD is 0


def _is_wiki(doc):
    if not doc.lstrip().startswith('= '):
        return False
    hits = 0
    for m in _WIKI_MARKERS:
        if m in doc:
            hits += 1
            if hits >= 2:
                return True
    return False


def _repetitive(words):
    """Existing repetition filter: max word frequency over first 500 words > 0.4."""
    if not words:
        return True
    sample_words = words if len(words) < 500 else words[:500]
    max_freq = max(Counter(sample_words).values())
    return max_freq / len(sample_words) > 0.4


def _score(doc, words, target_words):
    """The original relevance score, unchanged."""
    num_words = len(words)
    score = 0.0
    if doc.startswith('= '):
        score += 2.0
    if ' @-@ ' in doc:
        score += 1.5
    overlap_count = 0
    for w in words[:min(num_words, 200)]:
        if w in target_words:
            overlap_count += 1
    score += overlap_count / (num_words ** 0.5)
    if doc.count('.') > 2 and doc.count(',') > 1:
        score += 0.5
    return score


def _make_counter():
    """Returns a function mapping a list of texts to gpt2 token counts."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    def count(texts):
        if not texts:
            return []
        if enc is None:
            return [max(1, len(t) // 4) for t in texts]
        out = []
        chunk = 1024
        for s in range(0, len(texts), chunk):
            batch = enc.encode_ordinary_batch(texts[s:s + chunk], num_threads=8)
            out.extend(len(x) for x in batch)
        return out

    return count


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # ---- Target statistics -------------------------------------------------
    target_words = set()        # for the original score
    target_tokens = set()       # whitespace tokens, case kept
    target_bigrams = set()      # whitespace bigrams, case kept
    for ts in target_sample:
        target_words.update(re.findall(r'\w{4,}', ts.lower()))
        toks = ts.split()
        target_tokens.update(toks)
        target_bigrams.update(zip(toks, toks[1:]))

    # ---- Exact-duplicate removal (all docs) --------------------------------
    seen = set()
    unique = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        norm = " ".join(doc.lower().split())
        if not norm:
            continue
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(i)
    del seen

    # ---- Split into wiki-style / other -------------------------------------
    wiki_idx = []
    other_idx = []
    for i in unique:
        if _is_wiki(docs[i]):
            wiki_idx.append(i)
        else:
            other_idx.append(i)

    # ---- Damage statistics for wiki-style docs -----------------------------
    cand = []  # (idx, v, b)
    for i in wiki_idx:
        doc = docs[i]
        toks = doc.split()
        if len(toks) < _MIN_WIKI_TOKENS:
            continue
        if _repetitive(_WORD_RE.findall(doc[:20000].lower())):
            continue
        toks = toks[:_PREFIX_TOKENS]
        n = len(toks)
        v = sum(1 for t in toks if t in target_tokens) / n
        b = sum(1 for bg in zip(toks, toks[1:]) if bg in target_bigrams) / max(1, n - 1)
        cand.append((i, v, b))

    clean = []  # (idx, b)
    if cand:
        vs = np.array([c[1] for c in cand], dtype=np.float64)
        bs = np.array([c[2] for c in cand], dtype=np.float64)
        v_med = float(np.median(vs))
        b_med = float(np.median(bs))
        v_mad = max(float(np.median(np.abs(vs - v_med))), _MAD_FLOOR)
        b_mad = max(float(np.median(np.abs(bs - b_med))), _MAD_FLOOR)
        v_thr = v_med - 3.0 * v_mad
        b_thr = b_med - 3.0 * b_mad
        for i, v, b in cand:
            if v < v_thr or b < b_thr:
                continue
            clean.append((i, b))

    # ---- One doc per title line (keep the copy with the highest b) ---------
    best_by_title = {}
    for i, b in clean:
        first_line = docs[i].strip().split('\n', 1)[0]
        title = " ".join(first_line.split())
        prev = best_by_title.get(title)
        if prev is None or b > prev[1]:
            best_by_title[title] = (i, b)
    clean = sorted(best_by_title.values())  # by index, deterministic

    # ---- Token accounting --------------------------------------------------
    count_tokens = _make_counter()
    wiki_counts = count_tokens([docs[i] for i, _ in clean])
    W = sum(wiki_counts)

    selected = []
    if W > T:
        # Keep the highest-b docs until 1.0*T is reached.
        order = sorted(range(len(clean)), key=lambda k: (-clean[k][1], clean[k][0]))
        total = 0
        for k in order:
            if total >= T:
                break
            selected.append(clean[k][0])
            total += wiki_counts[k]
    elif W >= 0.6 * T:
        # Wiki only; the trainer cycles the list if it is short.
        selected = [i for i, _ in clean]
    else:
        # Top up with non-wiki docs in descending order of the original score.
        selected = [i for i, _ in clean]
        total = W
        scored = []
        for i in other_idx:
            doc = docs[i]
            if len(doc) < 50:
                continue
            words = _WORD_RE.findall(doc.lower())
            if len(words) < 10:
                continue
            if _repetitive(words):
                continue
            scored.append((_score(doc, words, target_words), i))
        scored.sort(key=lambda x: (-x[0], x[1]))

        chunk = 512
        pos = 0
        while total < T and pos < len(scored):
            batch = [i for _, i in scored[pos:pos + chunk]]
            pos += chunk
            counts = count_tokens([docs[i] for i in batch])
            for i, c in zip(batch, counts):
                if total >= T:
                    break
                selected.append(i)
                total += c

    if not selected:
        # Last-resort fallback so the trainer always has data.
        selected = unique if unique else list(range(len(docs)))
        selected = list(selected)

    rng.shuffle(selected)
    return selected
