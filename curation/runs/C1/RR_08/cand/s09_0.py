import os
import re
import math
import random
import zlib
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

_WORD_RE = re.compile(r"[a-z]+|[0-9]+|[^ \w]")
_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_CLEAN_RE = re.compile(r'[^a-z0-9]')

def _get_tokens(text):
    return _WORD_RE.findall(text.lower())

def _get_words(text):
    return _WORDS.findall(text[:4000].lower())

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    # 1. Token statistics
    tpc = 0.25
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        sample_idx = rng.sample(range(n), min(600, n))
        tok_sum = 0
        char_sum = 0
        for i in sample_idx:
            d = docs[i]
            if d:
                tok_sum += len(enc.encode(d, disallowed_special=()))
                char_sum += len(d)
        if char_sum > 0:
            tpc = tok_sum / char_sum
    except:
        pass

    # 2. Damaged-text filter
    bg_sample_idx = rng.sample(range(n), min(4000, n))
    bg_corpus = [docs[i] for i in bg_sample_idx]
    
    bigram_counts = Counter()
    def count_bigrams(text, counts_obj):
        tokens = _get_tokens(text)
        for i in range(len(tokens) - 1):
            counts_obj[(tokens[i], tokens[i+1])] += 1
            
    for t in target_sample:
        count_bigrams(t, bigram_counts)
    for b in bg_corpus:
        count_bigrams(b, bigram_counts)

    def get_hit_score(text, global_counts, personal_counts):
        tokens = _get_tokens(text)
        hits = 0
        for i in range(len(tokens) - 1):
            bg_cnt = global_counts[(tokens[i], tokens[i+1])]
            # Remove personal counts for leave-one-out
            adj_cnt = bg_cnt - personal_counts.get((tokens[i], tokens[i+1]), 0)
            if adj_cnt >= 3:
                hits += 1
        return hits / max(1, len(tokens) - 1)

    target_hits = []
    for t in target_sample:
        p_counts = Counter()
        count_bigrams(t, p_counts)
        target_hits.append(get_hit_score(t, bigram_counts, p_counts))
    
    target_hits.sort()
    thr = 0.6 * target_hits[max(0, len(target_hits) // 5 - 1)] if target_hits else 0.0
    
    intact_docs = []
    rest = []
    for i, d in enumerate(docs):
        if not d: continue
        p_counts = Counter()
        count_bigrams(d, p_counts)
        score = get_hit_score(d, bigram_counts, p_counts)
        if score < thr:
            rest.append(i)
        else:
            intact_docs.append(i)

    # 3. Unigram Scores
    tgt_counts = Counter()
    for t in target_sample:
        tgt_counts.update(_WORDS.findall(t))
    n_t = sum(tgt_counts.values())
    
    bg_counts = Counter()
    for i in bg_sample_idx:
        bg_counts.update(_WORDS.findall(docs[i]))
    n_b = sum(bg_counts.values())
    
    if n_t == 0 or n_b == 0:
        # Fallback to random if distributions are empty
        scores = [0.0] * n
    else:
        floor_b = 0.5 / n_b
        unk_w = math.log(1.0 - 0.7)
        weights = {}
        for w, c in bg_counts.items():
            pb = c / n_b
            pt = tgt_counts.get(w, 0) / n_t
            weights[w] = min(4.0, math.log((0.7 * pt + 0.3 * pb) / pb))
        for w, c in tgt_counts.items():
            if w not in weights:
                pt = c / n_t
                weights[w] = min(4.0, math.log((0.7 * pt + 0.3 * floor_b) / floor_b))
        
        scores = []
        for i, d in enumerate(docs):
            ws = _WORDS.findall(d[:4000].lower())
            if not ws:
                scores.append(0.0)
                continue
            s = sum(weights.get(w, unk_w) for w in ws)
            scores.append(s / len(ws))

    # Score intact documents and sort
    scored_intact = []
    for i in intact_docs:
        scored_intact.append((scores[i], i))
    scored_intact.sort(key=lambda x: x[0], reverse=True)

    # 4. Near-duplicate removal
    kept_docs = []
    seen_keys = set()
    sketch_map = {} # sketch_val -> list of doc_indices
    collision_counts = Counter() # doc_idx -> count of shared sketch values
    
    # To keep order of scoring but process sequentially
    for _, idx in scored_intact:
        d = docs[idx]
        # Key A
        raw_key_a = _CLEAN_RE.sub('', d[:300].lower())
        key_a = raw_key_a[:150]
        if len(key_a) >= 50 and key_a in seen_keys:
            continue
        
        # Sketch
        ws = _WORDS.findall(d[:4000].lower())
        shingles = []
        for j in range(len(ws) - 4):
            shingle = ' '.join(ws[j:j+5])
            shingles.append(zlib.crc32(shingle.encode()) & 0xFFFFFFFF)
        
        distinct_shingles = sorted(list(set(shingles)))
        if len(distinct_shingles) >= 12:
            sketch = sorted(distinct_shingles[:12])
            match_count = 0
            for s_val in sketch:
                if s_val in sketch_map:
                    match_count += 1
            
            if match_count >= 9:
                continue
            
            # Add to sketch maps
            for s_val in sketch:
                if s_val not in sketch_map:
                    sketch_map[s_val] = []
                sketch_map[s_val].append(idx)
        else:
            # Skip sketch check if < 12 shingles
            pass

        if len(key_a) >= 50:
            seen_keys.add(key_a)
        kept_docs.append(idx)

    # 5. Greedy fill
    total_tokens = T * 1.08
    current_tokens = 0
    final_selection = []
    
    # Use a set for O(1) lookup
    kept_set = set(kept_docs)
    
    # First, put kept docs in, but keep track of tokens
    # We need to sort kept_docs by score to ensure we pick best first
    scored_kept = sorted(scored_intact, key=lambda x: x[0], reverse=True)
    for s, idx in scored_kept:
        final_selection.append(idx)
        current_tokens += max(1.0, len(docs[idx]) * tpc)
        if current_tokens >= total_tokens:
            break
            
    if current_tokens < total_tokens:
        # Fill with rest
        rest_scored = []
        for i in rest:
            if i in kept_set: continue
            rest_scored.append((scores[i], i))
        rest_scored.sort(key=lambda x: x[0], reverse=True)
        
        for s, idx in rest_scored:
            final_selection.append(idx)
            current_tokens += max(1.0, len(docs[idx]) * tpc)
            if current_tokens >= total_tokens:
                break

    if not final_selection:
        final_selection = list(range(n))

    # 6. Shuffle
    rng.shuffle(final_selection)
    return final_selection
