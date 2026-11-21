```python
import os
import random
import zlib
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    N = len(docs)

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

    # Calibration setup
    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words:
            ref_words.add(w)
        for j in range(len(words) - 1):
            ref_bigrams.add(f"{words[j]} {words[j+1]}")

    # 1. Mask A
    unique_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        h = zlib.crc32(doc.replace(" ", "").encode('utf-8'))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

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

    mask_a = [False] * N
    doc_big_scores_a = {}
    
    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 1 - (len(set(lines)) / len(lines)) if lines else 0.0
        
        if len(words_all) < 40 or rep_val > 0.3:
            continue
            
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
            if i == 0 and not (zr_lo <= z <= zr_hi):
                all_pass = False; break
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct):
                all_pass = False; break
        
        if all_pass:
            mask_a[idx] = True
            doc_big_scores_a[idx] = sum(m[0] for m in win_metrics) / len(win_metrics)

    S_indices = [i for i, val in enumerate(mask_a) if val]
    
    # 2. Mask B
    mask_b = [False] * N
    cal_big3, cal_voc3, cal_punct3 = [], [], []
    for i in range(100, 200):
        doc = target_sample[i]
        pos = 0
        win_m_b, win_m_v, win_m_p = [], [], []
        while pos < len(doc):
            end = pos + 3000
            if end - pos < 1500 and end < len(doc): end = len(doc)
            win_text = doc[pos:end]
            b, v, p, z = get_metrics(win_text)
            win_m_b.append(b); win_m_v.append(v); win_m_p.append(p)
            pos = end
        cal_big3.append(min(win_m_b) if win_m_b else 1.0)
        cal_voc3.append(min(win_m_v) if win_m_v else 1.0)
        cal_punct3.append(min(win_m_p) if win_m_p else 1.0)
    
    cal_big3.sort(); cal_voc3.sort(); cal_punct3.sort()
    thr_b = 0.9 * cal_big3[int(5 * len(cal_big3) / 100)]
    thr_v = 0.97 * cal_voc3[int(5 * len(cal_voc3) / 100)]
    thr_p = 0.9 * cal_punct3[int(5 * len(cal_punct3) / 100)]

    # 3. Mask C
    cw = set()
    for i in range(100):
        for w in target_sample[i].split():
            cw.add(w)
    
    cal_cvoc, cal_up, cal_cap, cal_dig, cal_awl = [], [], [], [], []
    for i in range(100, 200):
        doc = target_sample[i][:6000]
        words = doc.split()
        if len(words) < 100: continue
        cvoc = sum(1 for w in words if w in cw) / len(words)
        up = sum(1 for char in doc if char.isupper()) / (sum(1 for char in doc if char.isalpha()) + 1)
        cap = sum(1 for w in words if w and w[0].isupper()) / len(words)
        dig = sum(1 for char in doc if char.isdigit()) / (len(doc) + 1)
        awl = sum(len(w) for w in words) / len(words)
        cal_cvoc.append(cvoc); cal_up.append(up); cal_cap.append(cap); cal_dig.append(dig); cal_awl.append(awl)
    
    cal_cvoc.sort(); cal_up.sort(); cal_cap.sort(); cal_dig.sort(); cal_awl.sort()
    thr_cvoc = 0.9 * cal_cvoc[int(5 * len(cal_cvoc) / 100)]
    up_lo, up_hi = 0.6 * cal_up[0], 1.5 * cal_up[-1]
    cap_lo, cap_hi = 0.6 * cal_cap[0], 1.5 * cal_cap[-1]
    dig_hi = 1.5 * cal_dig[-1]
    awl_lo, awl_hi = 0.9 * cal_awl[0], 1.1 * cal_awl[-1]

    # 4. Mask D calibration
    w2, b2 = set(), set()
    s_words = []
    for idx in S_indices:
        ws = docs[idx].lower().split()
        s_words.append(ws)
        for w in ws:
            if s_words.count(ws) > 1: # Logic intended: count distinct docs
                pass # Re-evaluating for speed:
    
    # Corrected D calibration logic
    docs_with_words = [set(docs[idx].lower().split()) for idx in S_indices]
    for i in range(len(docs_with_words)):
        for w in docs_with_words[i]:
            if sum(1 for s in docs_with_words if w in s) >= 2: w2.add(w)
    for i in range(len(S_indices)):
        ws = docs[S_indices[i]].lower().split()
        for j in range(len(ws)-1):
            pair = f"{ws[j]} {ws[j+1]}"
            if sum(1 for s in [docs[idx].lower().split() for idx in S_indices] if pair in s) >= 2: # Simplification for speed
                b2.add(pair)
    
    # Because the above is O(S^2), let's simplify to satisfy the "at least 2" rule efficiently
    w2_fast = set()
    b2_fast = set()
    doc_sets = [set(docs[i].lower().split()) for i in S_indices]
    for i in range(len(doc_sets)):
        for w in doc_sets[i]:
            w2_fast.add(w) # Placeholder - logic above is slow. 
    # Realized: Need better logic. Let's just use the sets.
    
    # Recalculate D calibration properly
    w2_fast = set()
    b2_fast = set()
    doc_sets = [set(docs[i].lower().split()) for i in S_indices]
    for i in range(len(doc_sets)):
        for w in doc_sets[i]:
            w2_fast.add(w)
    # (Due to complexity, simplified: build W2, B2 from S)
    # Actually, let's stick to the instruction: "lowercased words in at least 2 distinct docs of S"
    word_counts = {}
    for ds in doc_sets:
        for w in ds: word_counts[w] = word_counts.get(w, 0) + 1
    w2 = {w for w, c in word_counts.items() if c >= 2}
    
    # Bigrams B2
    bigram_counts = {}
    for i in range(len(S_indices)):
        ws = docs[S_indices[i]].lower().split()
        for j in range(len(ws)-1):
            pair = f"{ws[j]} {ws[j+1]}"
            bigram_counts[pair] = bigram_counts.get(pair, 0) + 1
    b2 = {pair for pair, c in bigram_counts.items() if c >= 2}

    # D calibration logic (actual)
    cal_min_big2, cal_min_voc2, cal_mean_big2 = [], [], []
    for idx in range(200):
        doc = target_sample[idx].lower().split()
        chunks = []
        curr = []
        for w in doc:
            curr.append(w)
            if len(curr) == 80:
                chunks.append(curr)
                curr = []
        if len(curr) >= 25:
            chunks.append(curr)
            curr = []
        elif curr:
            chunks[-1].extend(curr)
        
        m_b, m_v, m_mean = float('inf'), float('inf'), 0.0
        if not chunks: m_b, m_v = 1.0, 1.0
        for c in chunks:
            c_ws = c
            c_big = 0
            c_total = 0
            for i in range(len(c_ws)-1):
                pair = f"{c_ws[i]} {c_ws[i+1]}"
                if pair in b2: c_big += 1
                c_total += 1
            m_b = min(m_b, c_big / c_total if c_total > 0 else 1.0)
            m_v = min(m_v, sum(1 for w in c_ws if w in w2) / len(c_ws) if c_ws else 1.0)
            m_mean += (c_big / c_total if c_total > 0 else 0.0)
        
        cal_min_big2.append(m_b)
        cal_min_voc2.append(m_v)
        cal_mean_big2.append(m_mean / len(chunks) if chunks else 0.0)
    
    cal_min_big2.sort(); cal_min_voc2.sort(); cal_mean_big2.sort()
    thr_cb = 0.9 * cal_min_big2[int(5 * len(cal_min_big2) / 100)]
    thr_cv = 0.95 * cal_min_voc2[int(5 * len(cal_min_voc2) / 100)]
    thr_tm = 0.9 * cal_mean_big2[int(2 * len(cal_mean_big2) / 100)]

    # 5. Final Intersection
    final_indices = []
    doc_data = {} # idx -> (min_big2, min_voc2, mean_big2)

    for idx in range(N):
        if not mask_a[idx]: continue
        doc = docs[idx]
        ws = doc.lower().split()
        
        # Mask B check
        b_pass = True
        pos = 0
        while pos < len(doc):
            end = pos + 3000
            if end - pos < 1500 and end < len(doc): end = len(doc)
            win_text = doc[pos:end]
            b, v, p, z = get_metrics(win_text)
            if not (b >= thr_b and v >= thr_v and p >= thr_p):
                b_pass = False; break
            pos = end
        if not b_pass: continue
        
        # Mask C check
        c_pass = True
        pos = 0
        while pos < len(doc):
            chunk = doc[pos:pos+6000]
            c_ws = chunk.split()
            if len(c_ws) >= 100:
                cvoc = sum(1 for w in c_ws if w in cw) / len(c_ws)
                up = sum(1 for char in chunk if char.isupper()) / (sum(1 for char in chunk if char.isalpha()) + 1)
                cap = sum(1 for w in c_ws if w and w[0].isupper()) / len(c_ws)
                dig = sum(1 for char in chunk if char.isdigit()) / (len(chunk) + 1)
                awl = sum(len(w) for w in c_ws) / len(c_ws)
                if not (cvoc >= thr_cvoc and up_lo <= up <= up_hi and cap_lo <= cap <= cap_hi and dig <= dig_hi and awl_lo <= awl <= awl_hi):
