"""Curator: target-similarity filter + heuristics + dedup.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string
import hashlib
import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830,400

SPAM_MARKERS = ("click here to order", "best price guaranteed", "coupon")
_LETTERS = string.ascii_letters.encode()
_WS = b" \t\n\r\x0b\x0c"

def _ascii_letter_ratio(d):
    b = d.encode("utf-8", "replace")
    non_ws = len(b) - (len(b) - len(b.translate(None, _WS)))
    if non_ws <= 0:
        return 0.0
    letters = len(b) - len(b.translate(None, _LETTERS))
    return letters / non_ws

def _bonus(d):
    b = 0.0
    nl = d.find("\n")
    first = (d if nl < 0 else d[:nl]).strip()
    if first.startswith("= ") and first.endswith(" ="):
        b += 1.0
    if " @-@ " in d:
        b += 1.0
    if " , " in d or " . " in d:
        b += 0.5
    return b

def _get_stats(d):
    lines = d.splitlines()
    body = "\n".join(lines[1:]) if len(lines) > 0 else d
    text_chunk = body[:3000]
    tokens = text_chunk.split()
    
    s_br_count = 0
    s_br_total = 0
    valid_sentences = 0
    for i in range(len(lines)):
        line = lines[i]
        if line.strip().startswith("="):
            continue
        line_tokens = line.split()
        if len(line_tokens) > 5:
            last_t = line_tokens[-1]
            if not (last_t.endswith(('.', '"', ')', '!', '?', ';'))):
                valid_sentences += 1
        words_in_line = line.split()
        for j in range(len(words_in_line) - 1):
            if words_in_line[j].endswith('.'):
                s_br_total += 1
                next_t = words_in_line[j+1]
                if next_t and (next_t[0].isupper() or next_t[0].isdigit() or next_t[0] in ('"', '(', '[', '{')):
                    s_br_count += 1
    
    s_br = s_br_count / s_br_total if s_br_total > 0 else 0.0
    has_5_dots = s_br_total >= 5
    bi_list = ['of the', 'in the', 'to the', ', and', 'on the', 'at the', 'for the', 'by the', 'with the', 'from the', 'and the', ', the', 'as a', 'was a', 'to be', ', which', ', but']
    bi_count = 0
    for i in range(len(tokens) - 1):
        pair = " ".join([tokens[i], tokens[i+1]])
        if pair in bi_list:
            bi_count += 1
    bi_rate = (bi_count / (len(tokens) / 100.0)) if len(tokens) > 0 else 0.0
    has_150_tokens = len(tokens) >= 150
    return s_br, has_5_dots, bi_rate, has_150_tokens

def _hard_drop(d, target_stats=None):
    if not isinstance(d, str) or len(d) < 200:
        return True
    low = d.lower()
    for m in SPAM_MARKERS:
        if m in low:
            return True
    if _ascii_letter_ratio(d) < 0.6:
        return True
    if target_stats:
        s_br, has_5, bi_rate, has_150 = _get_stats(d)
        t_sbr5, t_bir5 = target_stats
        if has_5 and s_br < 0.5 * t_sbr5:
            return True
        if has_150 and bi_rate < 0.4 * t_bir5:
            return True
    return False

def _h(s):
    return hashlib.md5(s.encode("utf-8", "replace")).digest()

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    t_stats = []
    for t in target_sample:
        if isinstance(t, str):
            s_br, h5, bi_r, h150 = _get_stats(t)
            if h5: t_stats.append(s_br)
            if h150: t_stats.append(bi_r)
    
    t_sbr5 = np.percentile(t_stats[0::2], 5) if len(t_stats[0::2]) > 0 else 0.0
    t_bir5 = np.percentile(t_stats[1::2], 5) if len(t_stats[1::2]) > 0 else 0.0
    target_stats = (t_sbr5, t_bir5)

    keep = np.zeros(n, dtype=bool)
    seen_full, seen_prefix = set(), set()
    for i, d in enumerate(docs):
        if _hard_drop(d, target_stats): continue
        norm = " ".join(d.split())
        hf = _h(norm)
        if hf in seen_full: continue
        hp = _h(norm[:300])
        if hp in seen_prefix: continue
        seen_full.add(hf)
        seen_prefix.add(hp)
        keep[i] = True

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        targets = [t for t in target_sample if isinstance(t, str)]
        snip = int(max(300, min(2000, 24000000 // max(1, n + len(targets)))))
        pool_txt = [d[:snip] if isinstance(d, str) else "" for d in docs]
        tgt_txt = [t[:snip] for t in targets]
        
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, 
                               max_features=50000, dtype=np.float32)
        X = vec.fit_transform(pool_txt + tgt_txt)
        Xp, Xt = X[:n], X[n:]

        # Target mean bi-rate
        t_bi_rates = []
        for t in targets:
            _, _, br, _ = _get_stats(t)
            t_bi_rates.append(br)
        t_bi_mean = np.mean(t_bi_rates) if t_bi_rates else 1.0

        # Calculate quality score
        pool_bi_rates = []
        for i in range(n):
            if keep[i]:
                _, _, br, _ = _get_stats(docs[i])
                pool_bi_rates.append(br)
        
        # Filter by bi-rate 1.5x mean
        valid_indices = np.where(keep)[0]
        bi_rates_arr = np.array(pool_bi_rates)
        mask = (bi_rates_arr >= 0.5 * t_bi_mean) & (bi_rates_arr <= 1.5 * t_bi_mean)
        filtered_valid_indices = valid_indices[mask]

        if len(filtered_valid_indices) < len(valid_indices) * 0.3:
            # Fallback to Cosine Similarity if bi-rate filter is too aggressive
            scores = cosine_similarity(Xp, Xt.mean(axis=0)).flatten()
        else:
            # Use Cosine Similarity for ranking
            scores = cosine_similarity(Xp, Xt.mean(axis=0)).flatten()
            # Re-zero scores for docs not meeting bi-rate
            scores[np.setdiff1d(np.arange(n), filtered_valid_indices)] = -1.0
            
        # Final ranking logic
        kept_idx = np.where(keep)[0]
        kept_scores = scores[kept_idx]
        # Re-order scores to match kept_idx
        order = kept_idx[np.argsort(-kept_scores, kind="stable")]
    except Exception:
        kept_idx = np.where(keep)[0]
        order = kept_idx[np.random.permutation(len(kept_idx))]
        scores = np.zeros(n)

    for i in range(n):
        if keep[i]:
            scores[i] += _bonus(docs[i])
    for j, t in enumerate(targets):
        if j < len(scores): # simple safety
            scores[j] += _bonus(t)

    # Re-calculate scores for the order
    final_scores = np.array([scores[i] for i in order])
    
    # Filtering by threshold
    if len(targets) > 0:
        thr = 0.5 * (float(np.mean([scores[j] for j in range(len(targets))])) + float(np.median(final_scores)))
    else:
        thr = float(np.median(final_scores))
    
    n_in = int(np.sum(final_scores > thr))
    
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def count_batch(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts)]
    except:
        def count_batch(texts):
            return [max(1, len(x) // 4) for x in texts]

    target_tokens = int(T * 1.01)
    counts = []
    cum = 0
    pos = 0
    while pos < len(order) and cum < target_tokens:
        batch = [docs[int(i)] for i in order[pos:pos + 1000]]
        c = count_batch(batch)
        counts.extend(c)
        cum += sum(c)
        pos += 1000
    counts = np.asarray(counts, dtype=np.int64)
    csum = np.cumsum(counts)

    if n_counted := len(counts) > 0 and csum[-1] >= target_tokens:
        n_sel = int(np.searchsorted(csum, target_tokens, side="left")) + 1
        if n_in <= n_counted:
            in_tokens = int(csum[n_in - 1]) if n_in > 0 else 0
            if in_tokens < T and in_tokens >= 0.4 * T:
                n_sel = n_in
    else:
        n_sel = len(order)

    selected = [int(i) for i in order[:n_sel]]
    if not selected:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    rng.shuffle(selected)
    return selected
