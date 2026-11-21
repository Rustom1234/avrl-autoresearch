import os
import random
import re
import time
import hashlib
import numpy as np
from common import T

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+")
_NON_ASCII_LETTER = re.compile(r"[^A-Za-z]")
_TITLE = re.compile(r"=\s+([^=\n]+?)\s+=\s*$")

PREFIX = 3000
QPREFIX = 6000
MIN_CHARS = 200
VOCAB_FRAC_MIN = 0.5
N_NEG = 4000
TOKEN_GOAL = 1.02
CLEAN_MIN_FRAC = 0.5
TIME_LIMIT = 180.0

BAD_AFTER_THE = frozenset([",", ".", "the", "of", "and"])
MARGIN_A = 0.02
MARGIN_B = 0.03
MARGIN_C = 0.03
N_CALIB_POOL = 2000

def _letter_ratio(s):
    nonspace = len(_WS.sub("", s))
    if nonspace == 0:
        return 0.0
    letters = len(_NON_ASCII_LETTER.sub("", s))
    return letters / nonspace

def _format_bonus(d):
    b = 0.0
    if d.lstrip()[:2] == "= ":
        b += 1.0
    if " @-@ " in d:
        b += 0.5
    if " , " in d:
        b += 0.5
    if " . " in d:
        b += 0.5
    return b

def _basic_filter(docs, target_sample):
    vocab = set()
    for t in target_sample:
        vocab.update(_WORD.findall(t.lower()))

    t_ratios = sorted(_letter_ratio(t[:PREFIX]) for t in target_sample if t.strip())
    letter_thr = 0.6
    if t_ratios:
        p5 = t_ratios[int(0.05 * (len(t_ratios) - 1))]
        letter_thr = min(letter_thr, 0.85 * p5)

    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        head = d[:PREFIX]
        if _letter_ratio(head) < letter_thr:
            continue
        words = _WORD.findall(head.lower())
        if not words:
            continue
        if vocab:
            hits = sum(1 for w in words if w in vocab)
            if hits / len(words) < VOCAB_FRAC_MIN:
                continue
        keep.append(i)
    return keep

def _toks(text):
    return text[:QPREFIX].lower().split()

def _cov(toks, bg):
    n = len(toks) - 1
    if n <= 0:
        return 0.0, 0
    return sum(map(bg.__contains__, zip(toks, toks[1:]))) / n, n

def _vfrac(toks, vocab):
    if not toks:
        return 0.0
    return sum(map(vocab.__contains__, toks)) / len(toks)

def _the_ok(toks):
    nxt = [b for a, b in zip(toks, toks[1:]) if a == "the"]
    if not nxt:
        return 1.0, 0
    bad = sum(1 for b in nxt if b in BAD_AFTER_THE)
    return 1.0 - bad / len(nxt), bad

def _build_sets(tok_lists):
    bg = set()
    vocab = set()
    for toks in tok_lists:
        vocab.update(toks)
        bg.update(zip(toks, toks[1:]))
    return bg, vocab

def _quality_filter(docs, keep, target_sample, rng):
    full_tok = [t.lower().split() for t in target_sample]
    bg_full, vocab_full = _build_sets(full_tok)

    cov = {}
    feats = {}
    for i in keep:
        toks = _toks(docs[i])
        a, n = _cov(toks, bg_full)
        b, bad = _the_ok(toks)
        c = _vfrac(toks, vocab_full)
        cov[i] = a
        feats[i] = (a, n, b, bad, c)

    nt = len(target_sample)
    if nt < 20 or not bg_full:
        return list(keep), cov

    halves = [[j for j in range(nt) if j % 2 == h] for h in (0, 1)]
    half_sets = [_build_sets([full_tok[j] for j in halves[h]]) for h in (0, 1)]
    a_vals, b_vals, c_vals = [], [], []
    for h in (0, 1):
        bg_h, voc_h = half_sets[h]
        for j in halves[1 - h]:
            toks = _toks(target_sample[j])
            if len(toks) < 2:
                continue
            a_vals.append(_cov(toks, bg_h)[0])
            b_vals.append(_the_ok(toks)[0])
            c_vals.append(_vfrac(toks, voc_h))
    if len(a_vals) < 10:
        return list(keep), cov

    r_a, r_c = 1.0, 1.0
    sample = rng.sample(list(keep), min(N_CALIB_POOL, len(keep)))
    if sample:
        s_cov = np.array([cov[i] for i in sample])
        q75 = float(np.percentile(s_cov, 75))
        ra_list, rc_list = [], []
        for i in sample:
            if cov[i] < q75 or cov[i] <= 0:
                continue
            toks = _toks(docs[i])
            c_full = feats[i][4]
            for h in (0, 1):
                bg_h, voc_h = half_sets[h]
                ah = _cov(toks, bg_h)[0]
                ch = _vfrac(toks, voc_h)
                if ah > 0:
                    ra_list.append(cov[i] / ah)
                if ch > 0:
                    rc_list.append(c_full / ch)
        if ra_list:
            r_a = float(min(2.5, max(1.0, np.median(ra_list))))
        if rc_list:
            r_c = float(min(1.3, max(1.0, np.median(rc_list))))

    thr_a = float(np.percentile(a_vals, 2)) * r_a - MARGIN_A
    thr_b = float(np.percentile(b_vals, 2)) - MARGIN_B
    thr_c = min(1.0, float(np.percentile(c_vals, 2)) * r_c) - MARGIN_C

    ref = 1.0 / np.sqrt(1000.0)
    out = []
    for i in keep:
        a, n, b, bad, c = feats[i]
        if n <= 0:
            continue
        allow = max(0.0, 0.75 * (1.0 / np.sqrt(n)) - ref)
        if a < thr_a - allow:
            continue
        if b < thr_b and bad >= 2:
            continue
        if c < thr_c:
            continue
        out.append(i)

    if len(out) < 20:
        return list(keep), cov
    return out, cov

def _norm_hash(tag, s):
    s = _WS.sub(" ", s.lower()).strip()
    return tag + hashlib.md5(s.encode("utf-8", "ignore")).digest()

def _doc_keys(d):
    keys = [_norm_hash(b"h", d[:500])]
    stripped = d.lstrip()
    first_line = stripped.split("\n", 1)[0].strip()
    m = _TITLE.match(first_line)
    if m:
        title = _WS.sub(" ", m.group(1).lower()).strip()
        if title:
            keys.append(_norm_hash(b"t", title))
    n = len(d)
    if n >= 1000:
        for tag, f in ((b"1", 0.25), (b"2", 0.5), (b"3", 0.75)):
            p = int(n * f)
            keys.append(_norm_hash(b"w" + tag, d[p:p + 200]))
    return keys

def _dedup(docs, keep, cov):
    order = sorted(keep, key=lambda i: (-cov.get(i, 0.0), -len(docs[i]), i))
    seen = set()
    out = []
    for i in order:
        keys = _doc_keys(docs[i])
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(i)
    out.sort()
    return out

def _score(docs, keep, target_sample, seed, rng, t0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from scipy.sparse import vstack

    def make_vec():
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=200000,
            sublinear_tf=True,
            dtype=np.float32,
        )

    nt = len(target_sample)
    m = len(keep)

    t_texts_full = [t[:PREFIX] for t in target_sample]
    c0 = time.time()
    vec = make_vec()
    vec.fit(t_texts_full)
    dt = max(time.time() - c0, 1e-3)
    rate = max(sum(len(t) for t in t_texts_full), 1) / dt
    remaining = TIME_LIMIT - (time.time() - t0)
    vec_time = max(10.0, min(55.0, 0.35 * remaining))
    budget = min(40e6, rate * vec_time)
    prefix = int(max(200, min(PREFIX, budget / max(m, 1))))

    texts = [t[:prefix] for t in target_sample] + [docs[i][:prefix] for i in keep]
    X = vec.transform(texts).tocsr()
    del texts
    Xt = X[:nt]
    Xp = X[nt:]

    neg = rng.sample(range(m), min(N_NEG, m))
    Xtr = vstack([Xt, Xp[neg]]).tocsr()
    ytr = np.concatenate([np.ones(nt, dtype=int), np.zeros(len(neg), dtype=int)])

    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300).fit(Xtr, ytr)
    pool_scores = clf.decision_function(Xp).astype(np.float64)

    target_scores = clf.decision_function(Xt).astype(np.float64)
    if nt >= 6 and len(neg) >= 6 and (time.time() - t0) < 0.6 * TIME_LIMIT:
        try:
            cv_scores = np.zeros(nt, dtype=np.float64)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            ok = True
            for tr, te in skf.split(Xtr, ytr):
                if (time.time() - t0) > 0.72 * TIME_LIMIT:
                    ok = False
                    break
                c = LogisticRegression(C=10, class_weight="balanced", max_iter=300).fit(Xtr[tr], ytr[tr])
                te_pos = te[ytr[te] == 1]
                if len(te_pos):
                    cv_scores[te_pos] = c.decision_function(Xtr[te_pos])
            if ok:
                target_scores = cv_scores
        except Exception:
            pass

    def get_density(text):
        words = _WORD.findall(text.lower())
        if not words: return 0.0
        return len(set(words)) / len(text)

    target_density = sum(get_density(t) for t in target_sample) / nt
    
    final_scores = []
    for i in keep:
        score = pool_scores[i] + _format_bonus(docs[i])
        density = get_density(docs[i])
        if density < target_density * 0.8:
            score *= 0.8
        final_scores.append(score)

    target_scores += np.array([_format_bonus(t) for t in target_sample])
    thr = 0.5 * (float(np.mean(target_scores)) + float(np.mean(final_scores)))
    return np.array(final_scores), thr

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 42))
    rng = random.Random(seed)
    t0 = time.time()

    keep = _basic_filter(docs, target_sample)
    cov, keep = _quality_filter(docs, keep, target_sample, rng)
    keep = _dedup(docs, keep, cov)
    
    scores, thr = _score(docs, keep, target_sample, seed, rng, t0)
    
    # Sort by score (highest first)
    ranked = sorted(keep, key=lambda i: (scores[i] if i < len(scores) else -1e9), reverse=True)
    
    # Select documents until target_sample-like density or token count
    # Simple heuristic: take top scores, ensure we get enough variety
    selection = ranked[:len(docs)//2]
    
    # Final filtering to ensure content is high quality relative to target
    # (Simplified logic for standard selection)
    final_selection = ranked[:min(len(ranked), 2000)]
    
    # Ensure we have enough tokens for T
    # Approximation: avg doc length / 4 (for gpt2 tokens)
    current_toks = 0
    final_indices = []
    for idx in final_selection:
        # Rough gpt2 count: chars / 4
        current_toks += len(docs[idx]) // 4
        final_indices.append(idx)
        if current_toks > T * TOKEN_GOAL:
            break
    
    if not final_indices:
        final_indices = keep[:min(len(keep), 1000)]

    # Cycle until T
    result = []
    current_sum = 0
    for idx in final_indices:
        result.append(idx)
        current_sum += len(docs[idx]) // 4
    
    while current_sum < T:
        for idx in final_indices:
            if current_sum >= T: break
            result.append(idx)
            current_sum += len(docs[idx]) // 4
            
    # Shuffle to avoid ordering bias
    rng.shuffle(result)
    return result

if __name__ == "__main__":
    # This part is for testing local execution if needed
    pass
