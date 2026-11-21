import hashlib
import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from common import T

MIN_CHARS = 200
MAX_CHARS = 3000          # chars per doc used for scoring
CHAR_BUDGET = 30_000_000  # total chars vectorised
THRESHOLD = 0.5
TOK_BATCH = 1000

def _wiki_like(text):
    return text.lstrip().startswith("= ") and (" @-@ " in text or " , " in text)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # (1) length filter + exact dedup
    seen = set()
    valid = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        valid.append(i)

    if not valid:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(valid)
        rng.shuffle(out)
        return out

    # (2) char n-gram TF-IDF
    n = len(valid)
    # Ensure we don't exceed memory/budget
    cap = max(500, min(MAX_CHARS, CHAR_BUDGET // max(1, n)))
    pool_texts = [docs[i][:cap] for i in valid]
    target_texts = [t[:cap] for t in targets]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=50000,
        dtype=np.float32,
    )
    
    combined = pool_texts + target_texts
    vec.fit(combined)
    X_pool = vec.transform(pool_texts)
    X_pos = vec.transform(target_texts)

    # Calculate Cosine Similarity
    # dot_product[i, j] = X_pool[i] . X_pos[j]
    dot_product = X_pool.dot(X_pos.T).toarray()
    
    # Calculate row norms manually for sparse matrix
    def get_norms(matrix):
        # matrix.power(2) squares the values in the sparse matrix
        # sum(axis=1) gives the squared L2 norm per row
        return np.sqrt(np.array(matrix.power(2)).sum(axis=1))

    p_norms = get_norms(X_pool)
    t_norms = get_norms(X_pos)
    
    # Avoid division by zero
    p_norms[p_norms == 0] = 1.0
    t_norms[t_norms == 0] = 1.0
    
    # Cosine Sim matrix
    # Use broadcasting: (N_pool, 1) * (1, N_pos) -> (N_pool, N_pos)
    cos_sim = dot_product / (p_norms[:, np.newaxis] * t_norms)
    
    # Average of top 5 nearest neighbors
    # Sort each row of cos_sim descending
    sorted_sims = np.sort(cos_sim, axis=1)[..., ::-1]
    scores = np.mean(sorted_sims[:, :5], axis=1)

    # WikiText bonus
    target_frac = sum(1 for t in targets if _wiki_like(t)) / len(targets)
    bonus = 0.25 * target_frac
    if bonus > 0:
        flags = np.fromiter((_wiki_like(docs[i]) for i in valid), dtype=bool, count=n)
        scores = scores + bonus * flags

    # (3) Rank and keep
    order = np.argsort(-scores, kind="stable")
    candidates = [int(r) for r in order if scores[r] >= THRESHOLD]
    if not candidates:
        candidates = [int(r) for r in order]

    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total = 0
    done = False
    
    for start in range(0, len(candidates), TOK_BATCH):
        batch_indices = candidates[start:start + TOK_BATCH]
        batch_texts = [docs[valid[r]] for r in batch_indices]
        toks = enc.encode_ordinary_batch(batch_texts)
        
        for r, tk in zip(batch_indices, toks):
            kept.append(valid[r])
            total += len(tk)
            if total >= T:
                done = True
                break
        if done:
            break

    # (4) Seeded shuffle
    rng.shuffle(kept)
    return kept
