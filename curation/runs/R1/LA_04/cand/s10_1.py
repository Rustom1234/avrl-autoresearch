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
TIME_LIMIT = 180.0

BAD_AFTER_THE = frozenset([",", ".", "the", "of", "and"])
MARGIN_A = 0.02
MARGIN_B = 0.03
MARGIN_C = 0.03
N_CALIB_POOL = 2000

def _letter_ratio(s):
    nonspace = len(_WS.sub("", s))
    if nonspace == 0: return 0.0
    letters = len(_NON_ASCII_LETTER.sub("", s))
    return letters / nonspace

def _format_bonus(d):
    b = 0.0
    if d.lstrip()[:2] == "= ": b += 1.0
    if " @-@ " in d: b += 0.5
    if " , " in d: b += 0.5
    if " . " in d: b += 0.5
    return b

def _basic_filter(docs, target_sample):
    vocab = set()
    for t in target_sample:
        vocab.update(_WORD.findall(t.lower()))
    t_ratios = sorted(_letter_ratio(t[:PREFIX]) for t in target_sample if t.strip())
    letter_thr = 0.6
    if t_ratios:
        letter_thr = min(letter_thr, 0.85 * t_ratios[int(0.05 * (len(t_ratios) - 1))])
    
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS: continue
        head = d[:PREFIX]
        if _letter_ratio(head) < letter_thr: continue
        words = _WORD.findall(head.lower())
        if not words: continue
        if vocab:
            hits = sum(1 for w in words if w in vocab)
            if hits / len(words) < VOCAB_FRAC_MIN: continue
        keep.append(i)
    return keep

def _toks(text):
    return text[:QPREFIX].lower().split()

def _cov(toks, bg):
    n = len(toks) - 1
    if n <= 0: return 0.0, 0
    return sum(map(bg.__contains__, zip(toks, toks[1:]))) / n, n

def _vfrac(toks, vocab):
    if not toks: return 0.0
    return sum(map(vocab.__contains__, toks)) / len(toks)

def _the_ok(toks):
    nxt = [b for a, b in zip(toks, toks[1:]) if a == "the"]
    if not nxt: return 1.0, 0
    bad = sum(1 for b in nxt if b in BAD_AFTER_THE)
    return 1.0 - bad / len(nxt), bad

def _build_sets(tok_lists):
    bg, vocab = set(), set()
    for toks in tok_lists:
        vocab.update(toks)
        bg.update(zip(toks, toks[1:]))
    return bg, vocab

def _quality_filter(docs, keep, target_sample, rng):
    full_tok = [t.lower().split() for t in target_sample]
    bg_full, vocab_full = _build_sets(full_tok)
    cov, feats = {}, {}
    for i in keep:
        toks = _toks(docs[i])
        a, n = _cov(toks, bg_full)
        b, bad = _the_ok(toks)
        c = _vfrac(toks, vocab_full)
        cov[i] = a
        feats[i] = (a, n, b, bad, c)

    nt = len(target_sample)
    if nt < 20 or not bg_full: return list(keep), cov

    halves = [[j for j in range(nt) if j % 2 == h] for h in (0, 1)]
    half_sets = [_build_sets([full_tok[j] for j in halves[h]]) for h in (0, 1)]
    a_vals, b_vals, c_vals = [], [], []
    for h in (0, 1):
        bg_h, voc_h = half_sets[h]
        for j in halves[1 - h]:
            toks = _toks(target_sample[j])
            if len(toks) < 2: continue
            a_vals.append(_cov(toks, bg_h)[0])
            b_vals.append(_the_ok(toks)[0])
            c_vals.append(_vfrac(toks, voc_h))
    if len(a_vals) < 10: return list(keep), cov

    r_a, r_c = 1.0, 1.0
    sample = rng.sample(list(keep), min(N_CALIB_POOL, len(keep)))
    if sample:
        s_cov = np.array([cov[i] for i in sample])
        q75 = float(np.percentile(s_cov, 75))
        ra_list, rc_list = [], []
        for i in sample:
            if cov[i] < q75 or cov[i] <= 0: continue
            toks = _toks(docs[i])
            c_full = feats[i][4]
            for h in (0, 1):
                bg_h, voc_h = half_sets[h]
                ah = _cov(toks, bg_h)[0]
                ch = _vfrac(toks, voc_h)
                if ah > 0: ra_list.append(cov[i] / ah)
                if ch > 0: rc_list.append(c_full / ch)
        if ra_list: r_a = float(min(2.5, max(1.0, np.median(ra_list))))
        if rc_list: r_c = float(min(1.3, max(1.0, np.median(rc_list))))

    thr_a = float(np.percentile(a_vals, 2)) * r_a - MARGIN_A
    thr_b = float(np.percentile(b_vals, 2)) - MARGIN_B
    thr_c = min(1.0, float(np.percentile(c_vals, 2)) * r_c) - MARGIN_C
    ref = 1.0 / np.sqrt(1000.0)
    out = []
    for i in keep:
        a, n, b, bad, c = feats[i]
        if n <= 0: continue
        allow = max(0.0, 0.75 * (1.0 / np.sqrt(n)) - ref)
        if a < thr_a - allow or (b < thr_b and bad >= 2) or c < thr_c: continue
        d_full = docs[i]
        if len(d_full) > 2000:
            c_h = _cov(_toks(d_full[:2000]), bg_full)[0]
            c_m = _cov(_toks(d_full[len(d_full)//4 : len(d_full)//4 + 2000]), bg_full)[0]
            c_t = _cov(_toks(d_full[-2000:]), bg_full)[0]
            if np.std([c_h, c_m, c_t]) > 0.25: continue
        out.append(i)
    return out if len(out) >= 20 else list(keep), cov

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
        if title: keys.append(_norm_hash(b"t", title))
    n = len(d)
    if n >= 1000:
        for tag, f in ((b"1", 0.25), (b"2", 0.5), (b"3", 0.75)):
            p = int(n * f)
            keys.append(_norm_hash(b"w" + tag, d[p:p + 200]))
    return keys

def _dedup(docs, keep, cov):
    order = sorted(keep, key=lambda i: (-cov.get(i, 0.0), -len(docs[i]), i))
    seen, out = set(), []
    for i in order:
        keys = _doc_keys(docs[i])
        if any(k in seen for k in keys): continue
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
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=100000, sublinear_tf=True, dtype=np.float32)

    nt = len(target_sample)
    m = len(keep)
    t_texts_full = [t[:PREFIX] for t in target_sample]
    c0 = time.time()
    v = make_vec()
    v.fit_transform(t_texts_full)
    dt = max(time.time() - c0, 1e-3)
    rate = max(sum(len(t) for t in t_texts_full), 1) / dt
    remaining = TIME_LIMIT - (time.time() - t0)
    budget = min(20e6, rate * min(50.0, 0.3 * remaining))
    prefix = int(max(200, min(PREFIX, budget / max(m, 1))))

    texts = [t[:prefix] for t in target_sample] + [docs[i][:prefix] for i in keep]
    X = v.fit_transform(texts).tocsr()
    Xt, Xp = X[:nt], X[nt:]
    
    neg_idx = rng.sample(range(m), min(N_NEG, m))
    Xtr = vstack([Xt, Xp[neg_idx]]).tocsr()
    ytr = np.concatenate([np.ones(nt, dtype=int), np.zeros(len(neg_idx), dtype=int)])

    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300).fit(Xtr, ytr)
    pool_scores = clf.decision_function(Xp).astype(np.float64)
    target_scores = clf.decision_function(Xt).astype(np.float64)

    pool_scores += np.array([_format_bonus(docs[i]) for i in keep])
    target_scores += np.array([_format_bonus(t) for t in target_sample])
    return pool_scores, 0.5 * (np.mean(target_scores) + np.mean(pool_scores))

def select(docs, target_sample):
    seed = int(os.getenv("CURATE_SEED", 42))
    rng = random.Random(seed)
    keep = _basic_filter(docs, target_sample)
    keep, cov = _quality_filter(docs, keep, target_sample, rng)
    keep = _dedup(docs, keep, cov)
    
    if not keep: return [random.randint(0, len(docs)-1) for _ in range(int(T/2000))]
    
    pool_scores, thr = _score(docs, keep, target_sample, seed, rng, time.time())
    ranked = sorted(keep, key=lambda i: pool_scores[keep.index(i)], reverse=True)
    
    total_tokens = 0
    # Simplified token count for speed
    def approx_tokens(d): return len(d.split()) * 1.3
    
    selection = []
    for i in ranked:
        s_tokens = approx_tokens(docs[i])
        if total_tokens + s_tokens <= T:
            selection.append(i)
            total_tokens += s_tokens
        else: break
            
    if total_tokens < 0.3 * T:
        return selection
    return selection
