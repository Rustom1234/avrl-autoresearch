"""Curator: filter-then-shuffle. Keeps only clean, in-domain, unique documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. exact/near-duplicate removal (hash of first 2000 normalised characters)
  2. junk filter (length, zlib compression ratio, ASCII letter/space fraction)
  3. domain filter (TF-IDF Cosine Similarity to target_sample)
  4. shuffled-word damage filter (target bigram coverage, common bigram rate)
  5. seeded shuffle, with a top-up to 0.5 T tokens if too few documents survive
"""
import hashlib
import os
import random
import string
import zlib
import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover - fallback if common.py is unavailable
    T = 9_830_400

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.2
MIN_ALPHA_FRAC = 0.70
SCORE_CHARS = 3000
PROB_THRESHOLD = 0.5
SURFACE_RATE = 3.0          # (' , ' + ' . ') per 1000 characters
BIGRAM_WINDOW = 400         # first 400 word bigrams for target coverage
RATE_WINDOW = 2000          # words used for the common-bigram rate
MIN_COMMON_RATE = 1.0       # per 100 words
COVERAGE_FACTOR = 0.5       # times the median coverage of in-domain docs
TOPUP_FRACTION = 0.5        # of T

_ALPHA_SPACE = (string.ascii_letters + " ").encode("ascii")
_COMMON_BIGRAMS = {("of", "the"), ("in", "the"), (",", "and"), (".", "The")}

def _is_junk(d):
    """Step 2: too short, too repetitive, or too few letters/spaces."""
    n = len(d)
    if n < MIN_CHARS:
        return True
    b = d.encode("latin-1", "replace")
    non_alpha = len(b.translate(None, _ALPHA_SPACE))
    if (n - non_alpha) / n < MIN_ALPHA_FRAC:
        return True
    raw = d.encode("utf-8", "replace")
    if len(zlib.compress(raw)) / max(1, len(raw)) < MIN_ZLIB_RATIO:
        return True
    return False

def _surface_ok(d):
    """Wikitext surface form: heading start, or spaced punctuation."""
    if d.lstrip().startswith("= "):
        return True
    cnt = d.count(" , ") + d.count(" . ")
    return cnt * 1000.0 / max(1, len(d)) >= SURFACE_RATE

def _damage_stats(d, target_bigrams):
    """Step 4: (a) target-bigram coverage, (b) common-bigram rate per 100 words."""
    words = d.split(None, RATE_WINDOW)[:RATE_WINDOW]
    nw = len(words)
    if nw < 2:
        return 0.0, 0.0
    head = words[: BIGRAM_WINDOW + 1]
    nb = len(head) - 1
    hits = 0
    for bg in zip(head, head[1:]):
        if bg in target_bigrams:
            hits += 1
    cov = hits / nb
    common = 0
    for bg in zip(words, words[1:]):
        if bg in _COMMON_BIGRAMS:
            common += 1
    rate = common * 100.0 / nw
    return cov, rate

def _make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def fallback():
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    if n == 0:
        return []

    # ---- Steps 1 + 2: dedup (first occurrence wins), then junk filter ----
    seen = set()
    clean = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.lower().split())[:2000]
        key = hashlib.md5(norm.encode("utf-8", "replace")).digest()
        if key in seen:
            continue
        seen.add(key)
        if _is_junk(d):
            continue
        clean.append(i)
    del seen

    if not clean:
        return fallback()

    # ---- Step 3: Pointwise Ranking via Cosine Similarity ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Prepare target vectorizer
    # Use a subset of words (first 50) for similarity comparison as requested
    def get_50_words(text):
        return " ".join(text.split()[:50])

    target_texts = [get_50_words(t) for t in target_sample if isinstance(t, str)]
    if not target_texts:
        target_texts = ["dummy"]
        
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=True)
    target_tfidf = vec.fit_transform(target_texts)

    # Compute scores for clean documents
    clean_texts = [get_50_words(docs[i]) for i in clean]
    clean_tfidf = vec.transform(clean_texts)
    
    # Average cosine similarity to all 200 docs in target_sample
    # Resulting shape: (len(clean), 1)
    sim_matrix = cosine_similarity(clean_tfidf, target_tfidf)
    scores = np.mean(sim_matrix, axis=1)
    
    # Keep top 20% by score
    threshold_idx = int(len(scores) * 0.2)
    if threshold_idx > 0:
        top_indices = np.argsort(scores)[-threshold_idx:]
    else:
        top_indices = np.array([0])
        
    domain_ok_indices = [clean[k] for k in top_indices]
    # Apply secondary domain filter (surface)
    domain_ok = [i for i in domain_ok_indices if _surface_ok(docs[i])]

    # ---- Step 4: shuffled-word damage filter ----
    target_bigrams = set()
    for t in target_sample:
        if not isinstance(t, str):
            continue
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))

    cov_scores = np.zeros(len(domain_ok))
    rate_scores = np.zeros(len(domain_ok))
    for k, i in enumerate(domain_ok):
        cov_scores[k], rate_scores[k] = _damage_stats(docs[i], target_bigrams)

    med_cov = float(np.median(cov_scores)) if len(cov_scores) > 0 else 0.0
    
    keep_mask = (cov_scores >= COVERAGE_FACTOR * med_cov) & (rate_scores >= MIN_COMMON_RATE)
    survivors = [domain_ok[k] for k in np.nonzero(keep_mask)[0]]

    # ---- Step 5: token budget check, optional top-up, seeded shuffle ----
    target_tokens = int(TOPUP_FRACTION * T)
    count_tokens = _make_token_counter()
    BATCH = 512

    total = 0
    for s in range(0, len(survivors), BATCH):
        if total >= target_tokens:
            break
        batch = survivors[s : s + BATCH]
        total += sum(count_tokens([docs[i] for i in batch]))

    if total < target_tokens:
        # Documents that were high similarity but failed damage or surface
        # We take indices from clean that aren't in survivors, sorted by score
        remaining_in_clean = [i for i in clean if i not in survivors]
        # Scores for these are already calculated in clean_tfidf/scores
        # However, to stay simple and keep performance, we'll just filter the non-survivors
        # of the top 20% and sort them by score
        
        # Let's re-calculate scores for all clean to make ranking easier
        # (Optimization: we could have kept these, but for 20% selection it's cleaner)
        # We'll use the existing scores array for the indices in `clean`
        all_clean_scores = scores
        
        # Find indices in `clean` that are NOT in `survivors`
        surv_set = set(survivors)
        potential_rest = [i for i in clean if i not in surv_set]
        
        # Sort potential rest by score (descending)
        # We need to map potential_rest back to their positions in `clean` to get scores
        rank_map = {idx: k for k, idx in enumerate(clean)}
        potential_rest.sort(key=lambda x: all_clean_scores[rank_map[x]], reverse=True)
        
        done = False
        for i in potential_rest:
            survivors.append(i)
            total += count_tokens([docs[i]])[0]
            if total >= target_tokens:
                done = True
                break
            if done: break

    if not survivors:
        return fallback()

    rng.shuffle(survivors)
    return survivors
