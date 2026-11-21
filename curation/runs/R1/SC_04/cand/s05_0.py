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

    # 2. Calibration
    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words:
            ref_words.add(w)
        for j in range(len(words) - 1):
            ref_bigrams.add(f"{words[j]} {words[j+1]}")

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

    # 3. Score pool docs
    good_indices = []
    doc_big_scores = []
    
    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 0.0
        if lines:
            rep_val = 1 - (len(set(lines)) / len(lines))
        
        if len(words_all) < 40 or rep_val > 0.3:
            continue
            
        # Windowing
        windows = []
        pos = 0
        while pos < len(doc):
            end = pos + 6000
            if end - pos < 1500 and end < len(doc):
                end = len(doc)
            win_text = doc[pos:end]
            windows.append(win_text)
            pos = end
            if pos >= len(doc): break
        
        win_metrics = [get_metrics(w) for w in windows]
        
        all_pass = True
        for i, (b, v, p, z) in enumerate(win_metrics):
            if i == 0:
                if not (zr_lo <= z <= zr_hi):
                    all_pass = False; break
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct):
                all_pass = False; break
        
        if all_pass:
            good_indices.append(idx)
            doc_big_scores.append(sum(m[0] for m in win_metrics) / len(win_metrics))

    # 4. Title Dedupe
    titles = {}
    kept_indices = []
    title_map = {} # idx -> title_key
    
    # Separate titled vs non-titled
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
    
    # 5. Token Budget
    final_indices = non_titled
    token_counts = [len(enc.encode_ordinary(docs[i])) for i in final_indices]
    current_tokens = sum(token_counts)
    
    # Map of good_indices to their big scores for sorting
    big_score_map = {good_indices[i]: doc_big_scores[i] for i in range(len(good_indices))}
    
    if current_tokens > T:
        # Sort indices by big score asc to drop lowest
        sorted_by_big = sorted(final_indices, key=lambda x: big_score_map.get(x, 0))
        temp_tokens = current_tokens
        dropped = set()
        for idx in sorted_by_big:
            if temp_tokens <= T: break
            temp_tokens -= token_counts[final_indices.index(idx)]
            dropped.add(idx)
        final_indices = [i for i in final_indices if i not in dropped]
    elif current_tokens < 0.5 * T:
        # Add others that passed first-window (all good_indices)
        others = [i for i in good_indices if i not in final_indices]
        others.sort(key=lambda x: big_score_map.get(x, 0), reverse=True)
        for idx in others:
            if current_tokens >= 0.5 * T: break
            final_indices.append(idx)
            current_tokens += len(enc.encode_ordinary(docs[idx]))
            
    rng.shuffle(final_indices)
    return final_indices
