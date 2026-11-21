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

    # Stage 1 Pipeline
    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words:
            ref_words.add(w)
        for j in range(len(words) - 1):
            ref_bigrams.add(f"{words[j]} {words[j+1]}")

    cal_big = []
    cal_voc = []
    cal_punct = []
    cal_zr = []
    for i in range(100, 200):
        b, v, p, z = get_metrics(target_sample[i][:6000])
        cal_big.append(b); cal_voc.append(v); cal_punct.append(p); cal_zr.append(z)
    
    cal_big.sort(); cal_voc.sort(); cal_punct.sort(); cal_zr.sort()
    thr_big = 0.7 * cal_big[int(5 * len(cal_big) / 100)]
    thr_voc = 0.9 * cal_voc[int(5 * len(cal_voc) / 100)]
    thr_punct = 0.8 * cal_punct[int(5 * len(cal_punct) / 100)]
    zr_lo = 0.9 * cal_zr[0]
    zr_hi = 1.1 * cal_zr[-1]

    good_indices = []
    doc_big_scores = []
    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 1 - (len(set(lines)) / len(lines)) if lines else 0.0
        if len(words_all) < 40 or rep_val > 0.3: continue
        
        windows = []
        pos = 0
        while pos < len(doc):
            end = pos + 6000
            if end - pos < 1500 and end < len(doc): end = len(doc)
            windows.append(doc[pos:end])
            pos = end if pos < len(doc) else len(doc)
            if pos >= len(doc): break
        
        win_metrics = [get_metrics(w) for w in windows]
        all_pass = True
        for i, (b, v, p, z) in enumerate(win_metrics):
            if i == 0:
                if not (zr_lo <= z <= zr_hi): all_pass = False; break
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct): all_pass = False; break
        
        if all_pass:
            good_indices.append(idx)
            doc_big_scores.append(sum(m[0] for m in win_metrics) / len(win_metrics))

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
    
    S = non_titled

    # Stage 2 bootstrapping
    s_words_freq = {}
    s_bigrams_freq = {}
    for idx in S:
        doc = docs[idx]
        words = doc.lower().split()
        ws = set(words)
        for w in ws: s_words_freq[w] = s_words_freq.get(w, 0) + 1
        for j in range(len(words) - 1):
            pair = f"{words[j]} {words[j+1]}"
            s_bigrams_freq[pair] = s_bigrams_freq.get(pair, 0) + 1
    
    w2 = {w for w, count in s_words_freq.items() if count >= 2}
    b2 = {pair for pair, count in s_bigrams_freq.items() if count >= 2}
    
    # Calibration on S-only sets
    cal_w2 = w2
    cal_b2 = b2
    ref_words_2 = cal_w2
    ref_bigrams_2 = cal_b2

    def get_metrics_2(text):
        words = text.lower().split()
        bw_count = 0
        total_bw = 0
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            if pair in ref_bigrams_2: bw_count += 1
            total_bw += 1
        bv = bw_count / total_bw if total_bw > 0 else 0.0
        vv = sum(1 for w in words if w in ref_words_2) / len(words) if len(words) > 0 else 0.0
        return bv, vv

    cal_big2 = []
    cal_voc2 = []
    for doc in target_sample:
        text = doc
        windows = []
        pos = 0
        while pos < len(text):
            end = pos + 3000
            if end - pos < 1000 and end < len(text):
                end = len(text)
            windows.append(text[pos:end])
            pos = end if end < len(text) else len(text)
            if pos >= len(text): break
        
        # Each window cut at whitespace is slightly more complex but 
        # standard slicing is sufficient for this approximation
        for w in windows:
            b, v = get_metrics_2(w)
            cal_big2.append(b)
            cal_voc2.append(v)

    cal_big2.sort(); cal_voc2.sort()
    thr_big2 = 0.9 * (cal_big2[int(3 * len(cal_big2) / 100)])
    thr_voc2 = 0.97 * (cal_voc2[int(3 * len(cal_voc2) / 100)])

    # Stage 2 filter
    s2_indices = []
    s2_scores = []
    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 1 - (len(set(lines)) / len(lines)) if lines else 0.0
        if len(words_all) < 40 or rep_val > 0.3: continue
        
        # First window check (zr_lo, zr_hi from Stage 1)
        pos_z = 0
        end_z = pos_z + 6000
        if end_z - pos_z < 1500 and end_z < len(doc): end_z = len(doc)
        _, _, _, z_val = get_metrics(doc[pos_z:end_z])
        if not (zr_lo <= z_val <= zr_hi): continue
        
        # Windowed metrics
        windows_s2 = []
        pos_s = 0
        while pos_s < len(doc):
            end_s = pos_s + 3000
            if end_s - pos_s < 1000 and end_s < len(doc): end_s = len(doc)
            windows_s2.append(doc[pos_s:end_s])
            pos_s = end_s if end_s < len(doc) else len(doc)
            if pos_s >= len(doc): break
            
        passed_s2 = True
        sum_b2 = 0
        for w in windows_s2:
            b, v, p, _ = get_metrics(w) # Use Stage 1's get_metrics logic for p
            # But Stage 2 needs b2 and v2
            b2_val, v2_val = get_metrics_2(w)
            if b2_val < thr_big2 or v2_val < thr_voc2 or p < thr_punct:
                passed_s2 = False; break
            sum_b2 += b2_val
        
        if passed_s2:
            s2_indices.append(idx)
            s2_scores.append(sum_b2 / len(windows_s2))

    # Title dedupe on S2
    s2_titles = {}
    s2_non_titled = []
    for i, idx in enumerate(s2_indices):
        doc = docs[idx]
        first = doc.splitlines()[0] if doc else ""
        if first.startswith('='):
            key = first.strip().replace('=', '').strip()
            if key not in s2_titles: s2_titles[key] = []
            s2_titles[key].append((s2_scores[i], idx))
        else:
            s2_non_titled.append(idx)
    for key in s2_titles:
        s2_titles[key].sort(key=lambda x: x[0], reverse=True)
        s2_non_titled.append(s2_titles[key][0][1])

    # Token Budget
    token_counts = {}
    for idx in s2_indices:
        token_counts[idx] = len(enc.encode_ordinary(docs[idx]))
    
    current_tokens = sum(token_counts[idx] for idx in s2_non_titled)
    
    # Final selection
    if current_tokens > T:
        # Drop lowest mean-big2
        s2_non_titled.sort(key=lambda x: s2_scores[s2_indices.index(x)])
        dropped = set()
        temp_tokens = current_tokens
        for idx in s2_non_titled:
            if temp_tokens <= T: break
            temp_tokens -= token_counts[idx]
            dropped.add(idx)
        final_indices = [i for i in s2_non_titled if i not in dropped]
    elif current_tokens < 0.5 * T:
        final_indices = S
    else:
        final_indices = s2_non_titled

    rng.shuffle(final_indices)
    return final_indices
