"""Curation skills: the operations a curation pipeline is built from. Each one is a plain function.

Free skills (CPU): clean_lines, doc_ok, english_share, exact_dedup. GPU minutes: embed, semantic_dedup, target_score.
Model-powered: a judge gives scores to a sample (see gate1.py) and fit_scorer spreads them over the pool.
"""
import hashlib
import numpy as np

STOP = frozenset("the of and to a in is that for it as was with be by on not he this are or his from at which but "
                 "have an had they you".split())
EMB_MODEL = "BAAI/bge-small-en-v1.5"


def norm_line(ln):
    return " ".join(ln.split())


def clean_lines(text, common=frozenset()):
    """Line-level cleaning: drop short lines that are not sentences (menus, buttons, labels) and lines that many
    documents of the pool share (site boilerplate)."""
    out = []
    for ln in text.split("\n"):
        s = norm_line(ln)
        if len(s) < 30 and not s.endswith((".", "!", "?", ":", '"')):
            continue
        if s.lower() in common:
            continue
        out.append(s)
    return "\n".join(out)


def english_share(text):
    """Share of words that are common English function words. English prose is near 0.3, other languages under 0.1."""
    w = text.lower().split()
    return sum(x.strip(".,;:!?\"'()") in STOP for x in w) / max(len(w), 1)


def doc_ok(text, min_words=50, min_english=0.15):
    """Document-level heuristics in the style of the Gopher quality filter: length, word shape, symbols, language."""
    w = text.split()
    if len(w) < min_words:
        return False
    mean_len = sum(map(len, w)) / len(w)
    alpha = sum(any(c.isalpha() for c in x) for x in w) / len(w)
    symbols = (text.count("#") + text.count("...") + text.count("|")) / len(w)
    return 3 <= mean_len <= 10 and alpha >= 0.8 and symbols < 0.1 and english_share(text) >= min_english


def exact_dedup(texts):
    """Keep mask: the first copy of each document (whitespace and case ignored)."""
    seen, keep = set(), np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        h = hashlib.md5(norm_line(t).lower().encode()).digest()
        if h not in seen:
            seen.add(h)
            keep[i] = True
    return keep


def embed(texts, device="cuda", batch=256, max_tokens=512):
    """Unit-length sentence embeddings (bge-small), one row per text."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, device=device)
    m.max_seq_length = max_tokens
    return m.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def semantic_dedup(emb, thr=0.9, chunk=4096):
    """Keep mask: drop a document when an earlier one has cosine similarity above thr."""
    import torch
    e = torch.from_numpy(emb).cuda()
    keep = np.ones(len(emb), dtype=bool)
    for a in range(0, len(emb), chunk):
        s = e[a:a + chunk] @ e[:a + chunk].T
        rows = torch.arange(s.shape[0], device=s.device)[:, None]
        cols = torch.arange(s.shape[1], device=s.device)[None, :]
        s[cols >= rows + a] = -1                              # only earlier documents count
        keep[a:a + chunk] = (s.max(dim=1).values <= thr).cpu().numpy()
    return keep


def target_score(emb, target_emb, k=10):
    """Mean cosine similarity to the k nearest target documents."""
    s = emb @ target_emb.T
    return np.sort(s, axis=1)[:, -k:].mean(axis=1)


def fit_scorer(emb_labeled, y, emb_all):
    """Spread judge scores from a labeled sample to every document: ridge regression on the embeddings. The penalty
    is chosen by cross-validation on the labels alone. Returns (scores for emb_all, cross-validated R^2)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    m = RidgeCV(alphas=(0.1, 0.3, 1, 3, 10, 30))
    r2 = float(cross_val_score(m, emb_labeled, y, cv=5, scoring="r2").mean())
    return m.fit(emb_labeled, y).predict(emb_all), r2


def take_top(order, lens, budget):
    """The documents of `order` (best first) up to a token budget."""
    out, tot = [], 0
    for i in order:
        if tot >= budget:
            break
        out.append(int(i))
        tot += int(lens[i])
    return out
