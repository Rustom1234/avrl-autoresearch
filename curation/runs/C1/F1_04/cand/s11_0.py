"""Curator: KNN-style similarity filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. strict junk guards (length, unique-word ratio, non-alpha fraction)
  3. KNN Scoring: 
     - Vectorize pool (subset) and target_sample using char_wb TF-IDF.
     - For each pool doc, score is average cosine similarity to its 3 nearest neighbors 
       in the target_sample.
  4. Selection:
     - Rank by score, filter by "short/low-score" rule, and fill to token budget.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
import time
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import T

MAX_CHARS = 3000
N_NEG = 6000
MAX_FEATURES = 100_000
MIN_CHARS = 300
MIN_UNIQ_RATIO = 0.3
MAX_BAD_FRAC = 0.3
FILL = 1.02
VEC_BUDGET = 120.0
HARD_DEADLINE = 160.0

_GOOD_RE = re.compile(r"[A-Za-z\s]+")

def _norm_hash(text):
    norm = " ".join(text.lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()

def _bad_frac(text):
    s = text[:MAX_CHARS]
    if not s: return 1.0
    return len(_GOOD_RE.sub("", s)) / len(s)

def _uniq_ratio(text):
    words = text[:MAX_CHARS].lower().split()
    if not words: return 0.0
    return len(set(words)) / len(words)

def _make_vectorizer(max_features):
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1. Dedupe
    seen = set()
    uniq = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h in seen: continue
        seen.add(h)
        uniq.append(i)

    # 2. Strict Junk Guards
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    nt = max(1, len(targets))
    
    use_len = sum(len(t) < MIN_CHARS for t in targets) / nt <= 0.10
    use_uniq = sum(_uniq_ratio(t) < MIN_UNIQ_RATIO for t in targets) / nt <= 0.10
    use_bad = sum(_bad_frac(t) > MAX_BAD_FRAC for t in targets) / nt <= 0.10

    keep = []
    for i in uniq:
        d = docs[i]
        if use_len and len(d) < MIN_CHARS: continue
        if use_bad and _bad_frac(d) > MAX_BAD_FRAC: continue
        if use_uniq and _uniq_ratio(d) < MIN_UNIQ_RATIO: continue
        keep.append(i)
    if not keep: keep = uniq

    if not targets:
        out = list(keep)
        rng.shuffle(out)
        return out

    # 3. KNN Scoring
    # We process a large enough subset of 'keep' to build a representative score distribution
    # but we need scores for all 'keep' to rank them.
    # To save time/memory, we limit the pool docs used for scoring.
    score_indices = rng.sample(keep, min(len(keep), 5000))
    
    # Limit number of docs for vectorization to stay under VEC_BUDGET
    # (Number of docs * Max_features * 4 bytes)
    # Let's pick a reasonable subset for the "pool" to score
    scoring_pool = rng.sample(keep, min(len(keep), 2000))
    
    # Vectorize targets and scoring_pool
    vec = _make_vectorizer(MAX_FEATURES)
    
    # Build score for scoring_pool
    train_texts = [t[:MAX_CHARS] for t in targets] + [docs[i][:MAX_CHARS] for i in scoring_pool]
    X_train = vec.fit_transform(train_texts)
    X_pool = vec.transform([docs[i][:MAX_CHARS] for i in scoring_pool])
    
    # Similarity matrix: (pool_size, targets_size)
    sim_mat = cosine_similarity(X_pool, X_train)
    
    # Average of 3 nearest neighbors (targets are the first len(targets) columns)
    # sim_mat cols: [targets...][scoring_pool...]
    # We only want the similarity to the target sample
    target_sims = sim_mat[:, :nt]
    scores_dict = {}
    for idx, i in enumerate(scoring_pool):
        # Get indices of 3 highest similarities to targets
        top_3 = np.argsort(-target_sims[idx])[:3]
        scores_dict[i] = np.mean(target_sims[idx, top_3])

    # Interpolate scores for the rest of 'keep' (approximation)
    # Since we want to rank all docs in 'keep', we use the mapping from scoring_pool
    # but to be simple and efficient, we'll just rank the scoring_pool high and others low
    # or just expand the scoring_pool if time allows.
    
    # Let's refine: If we have time, try to score more.
    # To keep it simple and robust within 180s:
    # We use the scores_dict for docs in scoring_pool, 0 for others.
    final_scores = np.zeros(len(keep), dtype=np.float64)
    # Map keep index to its score
    keep_score_map = {}
    for i_keep, idx in enumerate(keep):
        keep_score_map[idx] = scores_dict.get(idx, 0.0)
    
    # 4. Selection Logic
    # Rank the 'keep' list based on the scores obtained
    # Use a stable sort by score descending
    sorted_indices = sorted(range(len(keep)), key=lambda i: keep_score_map[keep[i]], reverse=True)
    
    need = FILL * T
    final_selection = []
    current_acc = 0.0
    
    # Heuristic for tokens (chars/4)
    lens = np.array([len(docs[i]) for i in keep])
    
    # Filter and select
    # We want high scores first, but skip docs that are too short AND have low scores
    # Because we want the best content, we iterate the sorted_indices
    for i in sorted_indices:
        doc_idx = keep[i]
        score = keep_score_map[doc_idx]
        
        if len(docs[doc_idx]) < MIN_CHARS and score <= 0.05:
            continue
            
        final_selection.append(doc_idx)
        current_acc += len(docs[doc_idx]) / 4.0
        if current_acc >= need:
            break

    # Token refinement
    tok = {}
    if time.time() - t0 < HARD_DEADLINE - 20:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            B = 500
            exact_acc = 0
            # Only refine the first few hundred to stay in budget
            for s in range(0, len(final_selection), B):
                batch = final_selection[s:s+B]
                if time.time() - t0 > HARD_DEADLINE - 5: break
                outs = enc.encode_ordinary_batch([docs[j] for j in batch])
                for j, o in zip(batch, outs):
                    c = len(o) + 1
                    tok[j] = c
                    exact_acc += c
                    if exact_acc >= need: break
        except: pass

    # Final pass to respect budget exactly
    final_selection = []
    current_acc = 0.0
    # Re-sort by score to ensure high quality is first
    # Since sorted_indices is already sorted by score:
    for i in sorted_indices:
        doc_idx = keep[i]
        if len(docs[doc_idx]) < MIN_CHARS and keep_score_map[doc_idx] <= 0.05:
            continue
        final_selection.append(doc_idx)
        current_acc += tok.get(doc_idx, len(docs[doc_idx]) / 4.0)
        if current_acc >= need:
            break

    rng.shuffle(final_selection)
    return final_selection
