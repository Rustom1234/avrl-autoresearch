"""Target-similarity curator: classifier filter + dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. char (2-4, char_wb) + word (1-2) TF-IDF over pool + target_sample.
  2. LogisticRegression: target_sample = positives, seeded random 5000 pool docs = negatives;
     score every pool doc with decision_function.
  3. Dedup by hash of whitespace-normalised, lowercased first 500 chars (keep first).
  4. Drop junk (too short, low unique-word ratio, low alphabetic+space fraction).
  5. Rank survivors by score, take best until the cumulative gpt2 token count reaches T.
  6. Seeded shuffle of the selected indices.
"""
import hashlib
import os
import random

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from common import T

HEAD_CHARS = 3000       # max prefix used for features and junk statistics
MIN_FEAT_CHARS = 400    # lower bound of the adaptive feature prefix
FEAT_CHAR_BUDGET = 30_000_000  # total chars fed to the vectorizers (time bound)
DEDUP_CHARS = 500       # prefix used for the near-duplicate key
N_NEG = 5000            # Increased negative sample size
MIN_CHARS = 200
MIN_UNIQUE_RATIO = 0.2
MIN_ALPHA_FRAC = 0.7
BUDGET = 1.0 * T
TOK_CHUNK = 512


def _is_junk(doc):
    if len(doc) < MIN_CHARS:
        return True
    head = doc[:HEAD_CHARS]
    words = head.lower().split()
    if not words:
        return True
    if len(set(words)) / len(words) < MIN_UNIQUE_RATIO:
        return True
    good = sum(1 for c in head if c.isalpha() or c.isspace())
    if good / len(head) < MIN_ALPHA_FRAC:
        return True
    return False


def _dedup_key(doc):
    norm = " ".join(doc[:DEDUP_CHARS].lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _features(all_head):
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
        max_features=200000, dtype=np.float32,
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        max_features=200000, dtype=np.float32,
    )
    try:
        X = FeatureUnion([("char", char_vec), ("word", word_vec)]).fit_transform(all_head)
    except ValueError:  # empty word vocabulary (degenerate pool)
        X = char_vec.fit_transform(all_head)
    return X.tocsr()


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []
    m = len(target_sample)

    scores = np.zeros(n, dtype=np.float64)
    if m > 0:
        try:
            # ---- (1) features ---------------------------------------------------
            feat_chars = int(min(HEAD_CHARS, max(MIN_FEAT_CHARS, FEAT_CHAR_BUDGET // (n + m))))
            all_head = [d[:feat_chars] for d in docs] + [d[:feat_chars] for d in target_sample]
            X = _features(all_head)
            X_pool = X[:n]

            # ---- (2) classifier -------------------------------------------------
            neg_idx = np.asarray(
                random.Random(seed).sample(range(n), min(N_NEG, n)), dtype=np.int64
            )
            train_rows = np.concatenate([np.arange(n, n + m, dtype=np.int64), neg_idx])
            X_train = X[train_rows]
            y_train = np.concatenate(
                [np.ones(m, dtype=np.int32), np.zeros(len(neg_idx), dtype=np.int32)],
                axis=0
            )
            clf = LogisticRegression(
                C=10.0, class_weight="balanced", max_iter=1000, random_state=seed,
            )
            clf.fit(X_train, y_train)
            scores = np.asarray(clf.decision_function(X_pool), dtype=np.float64)
        except ValueError:
            scores = np.zeros(n, dtype=np.float64)

    # ---- (3) dedup + (4) junk filter ---------------------------------------
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if _is_junk(d):
            continue
        key = _dedup_key(d)
        if key in seen:
            continue
        seen.add(key)
        survivors.append(i)

    if not survivors:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    # ---- (5) rank by score, fill to the token budget ------------------------
    survivors = np.asarray(survivors, dtype=np.int64)
    order = survivors[np.argsort(-scores[survivors], kind="stable")]

    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total = 0
    done = False
    for start in range(0, len(order), TOK_CHUNK):
        chunk = order[start:start + TOK_CHUNK]
        toks = enc.encode_ordinary_batch([docs[int(i)] for i in chunk])
        for i, t in zip(chunk, toks):
            selected.append(int(i))
            total += len(t)
            if total >= BUDGET:
                done = True
                break
        if done:
            break

    # ---- (6) seeded shuffle -------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
