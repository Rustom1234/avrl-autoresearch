"""Curator: rank pool documents by ensemble of overlap, bigrams, and punctuation."""
import os
import random
import re
import time
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")
_BIGRAM_SHRINK = 20.0
_PUNCT_TOKENS = frozenset([",", ".", ";", ":", "!", "?"])
_PUNCT_PAIR_MAX = 0.03

def _token_counts(docs):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        counts = np.empty(len(docs), dtype=np.int64)
        chunk = 2000
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            try:
                encoded = enc.encode_ordinary_batch(batch, num_threads=8)
                lens = [len(e) for e in encoded]
            except Exception:
                lens = []
                for d in batch:
                    try:
                        lens.append(len(enc.encode_ordinary(d)))
                    except Exception:
                        lens.append(max(1, len(d) // 4))
            counts[start:start + len(batch)] = lens
        return counts
    except Exception:
        return np.array([max(1, len(d) // 4) for d in docs], dtype=np.int64)

def _overlap_scores(docs, target_sample):
    from sklearn.feature_extraction.text import CountVectorizer
    n = len(docs)
    vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN, dtype=np.float32)
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:
        return np.zeros(n, dtype=np.float64)
    word_re = re.compile(_TOKEN_PATTERN)
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(word_re.findall(t.lower()))
    vocab = vec.vocabulary_
    mask = np.zeros(X.shape[1], dtype=np.float64)
    for w in target_vocab:
        j = vocab.get(w)
        if j is not None:
            mask[j] = 1.0
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    X.data = np.log1p(X.data)
    num = np.asarray(X @ (idf * mask)).ravel()
    den = np.asarray(X @ idf).ravel()
    return num / (den + 30.0)

def _ws_tokens(text):
    return text.lower().split()

def _shuffled_tokens(text, rng):
    stripped = text.lstrip()
    head, sep, body = stripped.partition("\n")
    if sep:
        head_toks = _ws_tokens(head)
        body_toks = _ws_tokens(body)
    else:
        head_toks = []
        body_toks = _ws_tokens(stripped)
    rng.shuffle(body_toks)
    return head_toks + body_toks

def _calibrate(target_sample, seed):
    tok_lists = [_ws_tokens(t) for t in target_sample]
    doc_sets = [set(zip(toks, toks[1:])) for toks in tok_lists]
    doc_count = Counter()
    for s in doc_sets:
        doc_count.update(s)
    rng = random.Random(seed)
    real_scores = []
    shuf_scores = []
    for text, toks, own in zip(target_sample, tok_lists, doc_sets):
        nb = len(toks) - 1
        if nb <= 0: continue
        hits = sum(1 for b in zip(toks, toks[1:]) if doc_count[b] >= 2)
        real_scores.append(hits / (nb + _BIGRAM_SHRINK))
        stoks = _shuffled_tokens(text, rng)
        snb = len(stoks) - 1
        if snb <= 0: continue
        hits = sum(1 for b in zip(stoks, stoks[1:]) if doc_count.get(b, 0) and (doc_count[b] - (1 if b in own else 0)) >= 1)
        shuf_scores.append(hits / (snb + _BIGRAM_SHRINK))
    if not real_scores or not shuf_scores: return set(), 0.0
    lo_real = float(np.percentile(np.asarray(real_scores), 5))
    hi_shuf = float(np.percentile(np.asarray(shuf_scores), 95))
    return set(doc_count), 0.5 * (lo_real + hi_shuf)

def _order_stats(docs, bigram_set):
    n = len(docs)
    bigram_hit = np.zeros(n, dtype=np.float64)
    punct_frac = np.zeros(n, dtype=np.float64)
    contains = bigram_set.__contains__
    is_punct = _PUNCT_TOKENS.__contains__
    for i, d in enumerate(docs):
        toks = d.lower().split()
        nb = len(toks) - 1
        if nb <= 0: continue
        hits = sum(map(contains, zip(toks, toks[1:])))
        bigram_hit[i] = hits / (nb + _BIGRAM_SHRINK)
        flags = list(map(is_punct, toks))
        if any(flags):
            pairs = sum(1 for a, b in zip(flags, flags[1:]) if a and b)
            punct_frac[i] = pairs / nb
    return bigram_hit, punct_frac

def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    counts = _token_counts(docs)
    scores_a = _overlap_scores(docs, target_sample)
    bigram_set, threshold = _calibrate(target_sample, seed)
    bigram_hit, punct_frac = _order_stats(docs, bigram_set)

    score_b = None
    try:
        if time.time() - start_time < 120:
            pos_texts = []
            for t in target_sample:
                toks = _ws_tokens(t)
                for i in range(0, len(toks) - 299, 150):
                    pos_texts.append(" ".join(toks[i:i+300]))
                    break
                if not pos_texts: pos_texts.append(" ".join(toks[:300]))
            
            neg_indices = rng.sample(range(n), min(n, 6000))
            neg_texts = []
            for idx in neg_indices:
                t_toks = _ws_tokens(docs[idx])
                neg_texts.append(" ".join(t_toks[:300]))
            
            hv = HashingVectorizer(analyzer='word', token_pattern=r'\S+', 
                                   ngram_range=(1,2), n_features=2**18, 
                                   alternate_sign=False, norm='l2', binary=True)
            X_train = hv.transform(pos_texts + neg_texts)
            y_train = np.array([1]*len(pos_texts) + [0]*len(neg_texts))
            clf = LogisticRegression(C=1.0, class_weight='balanced', 
                                    solver='liblinear', max_iter=200).fit(X_train, y_train)
            
            score_b_vals = np.zeros(n)
            for i in range(n):
                t_toks = _ws_tokens(docs[i])
                if len(t_toks) < 300:
                    score_b_vals[i] = clf.decision_function(hv.transform([" ".join(t_toks)]))[0]
                else:
                    idx_list = [0, len(t_toks)//2, len(t_toks)-300]
                    vals = []
                    for start_idx in sorted(list(set(idx for idx in idx_list if idx < len(t_toks)-299))):
                        vals.append(clf.decision_function(hv.transform([" ".join(t_toks[start_idx:start_idx+300])]))[0])
                    score_b_vals[i] = np.mean(vals) if vals else 0.0
            score_b = score_b_vals
    except Exception:
        score_b = None

    punct_list = [',', '.', '@-@', '@,@', '@.@', '(', ')', ';', ':', '"', "'s", '=']
    score_c_vals = np.zeros(n)
    target_c_vec = np.zeros(len(punct_list))
    for t in target_sample:
        toks = _ws_tokens(t)
        t_vec = np.zeros(len(punct_list))
        for tok in toks:
            if tok == ',': t_vec[0] += 1/len(toks)
            elif tok == '.': t_vec[1] += 1/len(toks)
            elif tok == '@-@': t_vec[2] += 1/len(toks)
            elif tok == '@,@': t_vec[3] += 1/len(toks)
            elif tok == '@.@': t_vec[4] += 1/len(toks)
            elif tok == '(': t_vec[5] += 1/len(toks)
            elif tok == ')': t_vec[6] += 1/len(toks)
            elif tok == ';': t_vec[7] += 1/len(toks)
            elif tok == ':': t_vec[8] += 1/len(toks)
            elif tok == '"': t_vec[9] += 1/len(toks)
            elif tok == "'s": t_vec[10] += 1/len(toks)
            elif tok.endswith(',') or tok.endswith('.'): t_vec[11] += 1/len(toks)
        target_c_vec += t_vec
    target_c_vec /= len(target_sample)
    
    for i in range(n):
        toks = _ws_tokens(docs[i])
        d_vec = np.zeros(len(punct_list))
        for tok in toks:
            if tok == ',': d_vec[0] += 1/len(toks)
            elif tok == '.': d_vec[1] += 1/len(toks)
            elif tok == '@-@': d_vec[2] += 1/len(toks)
            elif tok == '@,@': d_vec[3] += 1/len(toks)
            elif tok == '@.@': d_vec[4] += 1/len(toks)
            elif tok == '(': d_vec[5] += 1/len(toks)
            elif tok == ')': d_vec[6] += 1/len(toks)
            elif tok == ';': d_vec[7] += 1/len(toks)
            elif tok == ':': d_vec[8] += 1/len(toks)
            elif tok == '"': d_vec[9] += 1/len(toks)
            elif tok == "'s": d_vec[10] += 1/len(toks)
            elif tok.endswith(',') or tok.endswith('.'): d_vec[11] += 1/len(toks)
        score_c_vals[i] = np.dot(d_vec, target_c_vec) / (np.linalg.norm(d_vec) * np.linalg.norm(target_c_vec) + 1e-9)

    def get_rank(arr):
        sorted_arr = sorted(arr, reverse=True)
        ranks = [1.0 - (i + 1) / len(sorted_arr) for i, val in enumerate(sorted(arr))]
        return np.array(ranks)

    if score_b is not None:
        rank_a = get_rank(scores_a * bigram_hit)
        rank_b = get_rank(score_b)
        rank_c = get_rank(score_c_vals)
        final_ranks = 0.4 * rank_b + 0.3 * rank_a + 0.3 * rank_c
    else:
        rank_a = get_rank(scores_a * bigram_hit)
        rank_c = get_rank(score_c_vals)
        final_ranks = 0.5 * rank_a + 0.5 * rank_c

    order_ok = bigram_hit >= threshold if bigram_set else np.ones(n, dtype=bool)
    punct_ok = punct_frac <= _PUNCT_PAIR_MAX
    
    seen = set()
    unique = np.ones(n, dtype=bool)
    for i, d in enumerate(docs):
        key = hash(_WS.sub(" ", d).strip().lower())
        if key in seen: unique[i] = False
        else: seen.add(key)

    usable = unique & (counts > 0) & order_ok & punct_ok
    
    titles = {}
    for i in range(n):
        if usable[i]:
            lines = docs[i].splitlines()
            if lines and lines[0].startswith('=') and lines[0].endswith('='):
                t = lines[0].lower().strip()
                if t not in titles or (bigram_hit[i] > titles[t][1]):
                    titles[t] = (i, bigram_hit[i])
    
    if titles:
        new_usable = np.zeros(n, dtype=bool)
        for t, (idx, _) in titles.items():
            new_usable[idx] = True
        usable = usable & new_usable

    if not usable.any():
        usable = unique & punct_ok
        if not usable.any(): usable = unique
        if not usable.any(): return list(range(n))

    rank_vals = final_ranks[usable]
    sorted_indices = np.argsort(-rank_vals)[::-1]
    usable_indices = np.where(usable)[0][sorted_indices]

    selected = []
    total_tokens = 0
    for i in usable_indices:
        selected.append(int(i))
        total_tokens += int(counts[i])
        if total_tokens >= T: break

    return selected
