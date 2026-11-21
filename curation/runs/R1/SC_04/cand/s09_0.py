import os
import random
import zlib
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # 1. Exact dedupe
    unique_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        h = zlib.crc32(doc.replace(" ", "").encode('utf-8'))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    def get_windows(text):
        windows = []
        pos = 0
        while pos < len(text):
            end = pos + 3000
            if end - pos < 3000:
                windows.append(text[pos:])
                break
            if end < len(text) and len(text) - end < 1500:
                windows.append(text[pos:])
                break
            windows.append(text[pos:end])
            pos = end
        # Logic correction: The prompt specifies: 
        # "cut the full doc at 3000-char steps. If the last piece is shorter than 1500 chars, 
        # merge it into the previous window, so every window is 1500 to 4500 chars. 
        # A doc shorter than 3000 chars is one window."
        
        # Re-implementing specific logic:
        res = []
        p = 0
        while p < len(text):
            e = p + 3000
            if e < len(text):
                rem = len(text) - e
                if rem < 1500:
                    res.append(text[p:e+rem])
                    break
                res.append(text[p:e])
                p = e
            else:
                res.append(text[p:])
                break
        return res

    def get_metrics(text):
        words = text.lower().split()
        bw_count = 0
        total_bw = 0
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            if pair in ref_bigrams:
                bw_count += 1
            total_bw += 1
        
        big_val = bw_count / total_bw if total_bw > 0 else 0.0
        voc_val = sum(1 for w in words if w in ref_words) / len(words) if len(words) > 0 else 0.0
        
        p1 = text.count(' , ')
        p2 = text.count(' . ')
        p3 = text.count(',')
        p4 = text.count('.')
        punct_val = (p1 + p2) / (p3 + p4 + 1)
        
        raw_bytes = text.encode('utf-8', errors='ignore')
        zr_val = len(zlib.compress(raw_bytes)) / len(raw_bytes) if len(raw_bytes) > 0 else 0.0
        
        return big_val, voc_val, punct_val, zr_val

    # Calibration Pre-processing
    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words:
            ref_words.add(w)
        for j in range(len(words) - 1):
            ref_bigrams.add(f"{words[j]} {words[j+1]}")

    cal_min_big = []
    cal_min_voc = []
    cal_min_punct = []
    
    # ZR calibration
    zr_vals = []
    for i in range(100, 200):
        win_ms = [get_metrics(w) for w in get_windows(target_sample[i])]
        # find min of each metric across windows
        cal_min_big.append(min(m[0] for m in win_ms))
        cal_min_voc.append(min(m[1] for m in win_ms))
        cal_min_punct.append(min(m[2] for m in win_ms))
        
    cal_min_big.sort()
    cal_min_voc.sort()
    cal_min_punct.sort()
    
    thr_big = 0.9 * cal_min_big[int(5 * len(cal_min_big) / 100)]
    thr_voc = 0.97 * cal_min_voc[int(5 * len(cal_min_voc) / 100)]
    thr_punct = 0.9 * cal_min_punct[int(5 * len(cal_min_punct) / 100)]

    # ZR bounds
    zr_vals = []
    for i in range(100, 200):
        _, _, _, z = get_metrics(target_sample[i][:6000])
        zr_vals.append(z)
    zr_vals.sort()
    zr_lo = 0.9 * zr_vals[0]
    zr_hi = 1.1 * zr_vals[-1]

    # Score pool docs
    good_indices = []
    doc_big_scores = []
    
    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 1 - (len(set(lines)) / len(lines)) if lines else 0.0
        
        if len(words_all) < 40 or rep_val > 0.3:
            continue
            
        windows = get_windows(doc)
        win_metrics = [get_metrics(w) for w in windows]
        
        all_pass = True
        # Check ZR on first window (effectively doc start)
        _, _, _, z = win_metrics[0]
        if not (zr_lo <= z <= zr_hi):
            all_pass = False
        
        if all_pass:
            for b, v, p, z in win_metrics:
                if not (b >= thr_big and v >= thr_voc and p >= thr_punct):
                    all_pass = False
                    break
        
        if all_pass:
            good_indices.append(idx)
            doc_big_scores.append(sum(m[0] for m in win_metrics) / len(win_metrics))

    # Title Dedupe
    titles = {}
    non_titled = []
    for i, idx in enumerate(good_indices):
        doc = docs[idx]
        first = doc.splitlines()[0] if doc else ""
        if first.startswith('='):
            key = first.strip().replace('=', '').strip()
            if key not in titles: titles[key] = []
            titles[key].append((doc_big_scores[i], idx))
        else:
            non_titled.append(idx)
            
    for key in titles:
        titles[key].sort(key=lambda x: x[0], reverse=True)
        non_titled.append(titles[key][0][1])
    
    # Token Budget
    final_indices = non_titled
    # Pre-calculate tokens
    token_map = {}
    for i in good_indices:
        token_map[i] = len(enc.encode_ordinary(docs[i]))
        
    current_tokens = sum(token_map[i] for i in final_indices)
    
    big_score_map = {good_indices[i]: doc_big_scores[i] for i in range(len(good_indices))}
    
    if current_tokens > T:
        sorted_by_big = sorted(final_indices, key=lambda x: big_score_map.get(x, 0))
        temp_tokens = current_tokens
        dropped = set()
        for idx in sorted_by_big:
            if temp_tokens <= T: break
            temp_tokens -= token_map.get(idx, 0)
            dropped.add(idx)
        final_indices = [i for i in final_indices if i not in dropped]
    elif current_tokens < 0.5 * T:
        others = [i for i in good_indices if i not in final_indices]
        others.sort(key=lambda x: big_score_map.get(x, 0), reverse=True)
        for idx in others:
            if current_tokens >= 0.5 * T: break
            final_indices.append(idx)
            current_tokens += token_map.get(idx, 0)
            
    rng.shuffle(final_indices)
    return final_indices
