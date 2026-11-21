"""Curator: target-similarity filter + enhanced damage detection + dedup.

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
    T = 9_830_400

SPAM_MARKERS = ("click here to order", "best price guaranteed", "coupon")
_LETTERS = string.ascii_letters.encode()
_WS = b" \t\n\r\x0b\x0c"

_CHAR_BUDGET = 24_000_000

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
    for line in lines:
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

    bi_list = {'of the', 'in the', 'to the', ', and', 'on the', 'at the', 'for the', 'by the', 'with the', 'from the', 'and the', ', the', 'as a', 'was a', 'to be', ', which', ', but'}
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

    # 1. Target Stats & Window Stats
    t_sbr_list, t_bir_list = [], []
    all_windows = []
    
    bi_list = {'of the', 'in the', 'to the', ', and', 'on the', 'at the', 'for the', 'by the', 'with the', 'from the', 'and the', ', the', 'as a', 'was a', 'to be', ', which', ', but'}
    
    def get_window_stats(text):
        lines = text.splitlines()
        body = "\n".join(lines[1:]) if len(lines) > 0 else text
        tokens = body.split()
        ws = []
        for i in range(0, len(tokens), 250):
            win = tokens[i:i+250]
            if len(win) < 120: break
            bi_c = 0
            for j in range(len(win)-1):
                if " ".join([win[j], win[j+1]]) in bi_list:
                    bi_c += 1
            bi_r = (bi_c / (len(win) / 100.0)) if len(win) > 0 else 0.0
            
            win_bc = 0
            win_bt = 0
            for j in range(len(win)-1):
                if win[j].endswith('.'):
                    win_bt += 1
                    nt = win[j+1]
                    if nt and (nt[0].isupper() or nt[0].isdigit() or nt[0] in ('"', '(', '[', '{')):
                        win_bc += 1
            s_br = win_bc / win_bt if win_bt > 0 else 0.0
            if win_bt >= 4:
                ws.append((bi_r, s_br))
        return ws

    for t in target_sample:
        if isinstance(t, str):
            s_br, h5, bi_r, h150 = _get_stats(t)
            if h5: t_sbr_list.append(s_br)
            if h150: t_bir_list.append(bi_r)
            ws_t = get_window_stats(t)
            all_windows.extend(ws_t)

    t_sbr5 = np.percentile(t_sbr_list, 5) if t_sbr_list else 0.0
    t_bir5 = np.percentile(t_bir_list, 5) if t_bir_list else 0.0
    
    all_ws = np.array(all_windows)
    b2 = np.percentile(all_ws[:,0], 2) if all_ws.size > 0 else 0.0
    s2 = np.percentile(all_ws[:,1], 2) if all_ws.size > 0 else 0.0

    keep = np.zeros(n, dtype=bool)
    vocab = set()
    pool_counts = {}
    for d in docs:
        if not isinstance(d, str): continue
        words = d.lower().split()
        for w in words:
            if w.isalpha() and len(w) >= 4:
                pool_counts[w] = pool_counts.get(w, 0) + 1
    for t in target_sample:
        if isinstance(t, str):
            words = t.lower().split()
            for w in words:
                if w.isalpha() and len(w) >= 4:
                    vocab.add(w)
    for w, count in pool_counts.items():
        if count >= 3: vocab.add(w)
    
    target_vocab_99 = 0
    target_max_vo = 0
    if target_sample:
        vo_list = []
        for t in target_sample:
            if not isinstance(t, str): continue
            words = t.lower().split()
            count = 0
            total = 0
            for w in words:
                if w.isalpha() and len(w) >= 4:
                    total += 1
                    if w in vocab: count += 1
            if total >= 100:
                vo_list.append(count / total)
        if vo_list:
            target_vocab_99 = np.percentile(vo_list, 99)
            target_max_vo = max(vo_list)

    for i, d in enumerate(docs):
        if _hard_drop(d, (t_sbr5, t_bir5)): continue
        ws = get_window_stats(d)
        bad_count = 0
        consecutive_bad = 0
        for w in ws:
            if w[0] < 0.6 * b2 or w[1] < 0.7 * s2:
                bad_count += 1
                consecutive_bad += 1
            else:
                consecutive_bad = 0
        if (len(ws) > 0 and bad_count / len(ws) > 0.25) or consecutive_bad >= 2:
            continue
            
        words = d.lower().split()
        vo_count = 0
        vo_total = 0
        for w in words:
            if w.isalpha() and len(w) >= 4:
                vo_total += 1
                if w in vocab: vo_count += 1
        if vo_total >= 100:
            frac = vo_count / vo_total
            if frac > max(2 * target_vocab_99, target_max_vo + 0.03):
                continue
                
        lines = [l for l in d.splitlines() if len(l) > 40]
        if len(lines) > 0 and len(set(lines)) / len(lines) < 0.6:
            continue
        keep[i] = True

    if not keep.any():
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
        
    sorted_indices = np.argsort(-[len(docs[i]) for i in range(n)])
    kept_indices = []
    seen_hashes = set()
    for i in sorted_indices:
        if not keep[i]: continue
        d = docs[i]
        norm = " ".join(d.split())
        h_full = _h(norm)
        h_pre = _h(norm[:300])
        h_mid = _h(norm[1000:1400]) if len(norm) >= 1400 else None
        h_last = _h(norm[-300:]) if len(norm) >= 300 else None
        if h_full in seen_hashes: continue
        if h_pre in seen_hashes: continue
        if h_mid is not None and h_mid in seen_hashes: continue
        if h_last is not None and h_last in seen_hashes: continue
        seen_hashes.add(h_full)
        seen_hashes.add(h_pre)
        if h_mid: seen_hashes.add(h_mid)
        if h_last: seen_hashes.add(h_last)
        kept_indices.append(i)

    targets = [t for t in target_sample if isinstance(t, str)]
    scores = np.zeros(n, dtype=np.float64)
    tgt_scores = np.zeros(len(targets), dtype=np.float64)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import vstack
        snip = int(max(300, min(2000, _CHAR_BUDGET // max(1, n + len(targets)))))
        pool_txt = [docs[i][:snip] if isinstance(docs[i], str) else "" for i in range(n)]
        tgt_txt = [t[:snip] for t in targets]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, max_features=50000, dtype=np.float32)
        X = vec.fit_transform(pool_txt + tgt_txt)
        Xp, Xt = X[:n], X[n:]
        neg_idx = rng.sample(range(n), min(3000, n))
        if len(targets) > 0 and len(neg_idx) > 0:
            Xtr = vstack([Xt, Xp[neg_idx]])
            y = np.concatenate([np.ones(Xt.shape[0]), np.zeros(len(neg_idx))])
            clf = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", random_state=seed)
            clf.fit(Xtr, y)
            scores = np.asarray(clf.decision_function(Xp), dtype=np.float64)
            tgt_scores = np.asarray(clf.decision_function(Xt), dtype=np.float64)
    except: pass

    for i in kept_indices:
        scores[i] += _bonus(docs[i])
    for j, t in enumerate(targets):
        tgt_scores[j] += _bonus(t)

    kept_scores = scores[kept_indices]
    order = np.array(kept_indices)[np.argsort(-kept_scores, kind="stable")]
    
    if len(targets) > 0:
        thr = 0.5 * (float(np.mean(tgt_scores)) + float(np.median(kept_scores)))
    else:
        thr = float(np.median(kept_scores))
    
    final_order = []
    for idx in order:
        if scores[idx] >= thr:
            final_order.append(idx)
    
    if not final_order:
        final_order = list(order)
        
    return final_order
