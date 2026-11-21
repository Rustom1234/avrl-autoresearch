import os
import random
import zlib
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # --- Original Logic Pre-requisites ---
    unique_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        h = zlib.crc32(doc.replace(" ", "").encode('utf-8'))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    # --- Stage 1: Original Pipeline ---
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

    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words: ref_words.add(w)
        for j in range(len(words) - 1): ref_bigrams.add(f"{words[j]} {words[j+1]}")

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
            pos = end
            if pos >= len(doc): break
        win_metrics = [get_metrics(w) for w in windows]
        all_pass = True
        for i, (b, v, p, z) in enumerate(win_metrics):
            if i == 0 and not (zr_lo <= z <= zr_hi): all_pass = False; break
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct): all_pass = False; break
        if all_pass:
            good_indices.append(idx)
            doc_big_scores.append(sum(m[0] for m in win_metrics) / len(win_metrics))

    s_indices = []
    titles_s = {}
    non_titled_s = []
    for i, idx in enumerate(good_indices):
        doc = docs[idx]
        first = doc.splitlines()[0] if doc else ""
        if first.startswith('='):
            key = first.strip().replace('=', '').strip()
            if key not in titles_s: titles_s[key] = []
            titles_s[key].append((doc_big_scores[i], idx))
        else:
            non_titled_s.append(idx)
    for key in titles_s:
        titles_s[key].sort(key=lambda x: x[0], reverse=True)
        non_titled_s.append(titles_s[key][0][1])
    s_indices = non_titled_s

    # --- Stage 2: Bootstrapped Second Stage ---
    w2_set = set()
    b2_set = set()
    doc_w_sets = []
    for idx in s_indices:
        words = docs[idx].lower().split()
        ws = set(words)
        doc_w_sets.append(ws)
        for i in range(len(words)-1):
            b2_set.add(f"{words[i]} {words[i+1]}")
        w2_set.update(ws)
    
    # Re-identify w2/b2 (occur in >= 2 docs)
    w2_set = set()
    b2_set = set()
    # Use local sets to count frequencies
    word_freqs = {}
    bigram_freqs = {}
    for ws in doc_w_sets:
        for w in ws: word_freqs[w] = word_freqs.get(w, 0) + 1
        # For bigrams, we need them from actual docs
        # But wait, we need the actual bigrams from the docs again
        # To stay efficient, we process docs in s_indices
    
    # Let's do it correctly:
    w2_set = set()
    b2_set = set()
    word_counts = {}
    bigram_counts = {}
    for idx in s_indices:
        words = docs[idx].lower().split()
        seen_words_in_doc = set()
        for w in words:
            if w not in seen_words_in_doc:
                word_counts[w] = word_counts.get(w, 0) + 1
                seen_words_in_doc.add(w)
        for i in range(len(words)-1):
            bi = f"{words[i]} {words[i+1]}"
            bigram_counts[bi] = bigram_counts.get(bi, 0) + 1
    
    for w, count in word_counts.items():
        if count >= 2: w2_set.add(w)
    for bi, count in bigram_counts.items():
        if count >= 2: b2_set.add(bi)

    def get_chunk_metrics(doc_words):
        chunks = []
        i = 0
        while i < len(doc_words):
            chunk = doc_words[i : i+80]
            if len(chunk) >= 80:
                chunks.append(chunk)
                i += 80
            else:
                rem = doc_words[i:]
                if len(rem) >= 25:
                    chunks.append(rem)
                    i = len(doc_words)
                else:
                    if chunks:
                        chunks[-1].extend(rem)
                        i = len(doc_words)
                    else:
                        chunks.append(rem)
                        i = len(doc_words)
        
        res_b = []
        res_v = []
        for c in chunks:
            cb = 0
            cv = 0
            for j in range(len(c)):
                if j > 0:
                    bi = f"{c[j-1]} {c[j]}"
                    if bi in b2_set: cb += 1
                if c[j] in w2_set: cv += 1
            res_b.append(cb / (len(c)-1) if len(c) > 1 else 0.0)
            res_v.append(cv / len(c) if len(c) > 0 else 0.0)
        
        min_b = min(res_b) if res_b else 0.0
        min_v = min(res_v) if res_v else 0.0
        mean_b = sum(res_b) / len(res_b) if res_b else 0.0
        return min_b, min_v, mean_b

    # Calibrate Stage 2
    cal_min_b = []
    cal_min_v = []
    cal_mean_b = []
    for i in range(200):
        w_list = target_sample[i].lower().split()
        mb, mv, mb_mean = get_chunk_metrics(w_list)
        cal_min_b.append(mb); cal_min_v.append(mv); cal_mean_b.append(mb_mean)
    
    cal_min_b.sort(); cal_min_v.sort(); cal_mean_b.sort()
    thr_cb = 0.9 * cal_min_b[int(5 * len(cal_min_b) / 100)]
    thr_cv = 0.95 * cal_min_v[int(5 * len(cal_min_v) / 100)]
    thr_mean = 0.9 * cal_mean_b[int(2 * len(cal_mean_b) / 100)]

    # Stage 2 Selection
    s2_candidates = []
    s2_scores = {}
    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 1 - (len(set(lines)) / len(lines)) if lines else 0.0
        if len(words_all) < 40 or rep_val > 0.3: continue
        
        # Punctuation and ZR checks
        valid_win = True
        pos = 0
        while pos < len(doc):
            end = pos + 6000
            if end - pos < 1500 and end < len(doc): end = len(doc)
            win_text = doc[pos:end]
            _, _, p_val, z_val = get_metrics(win_text)
            if pos == 0:
                if not (zr_lo <= z_val <= zr_hi): valid_win = False; break
            if not (p_val >= thr_punct): valid_win = False; break
            pos = end
            if pos >= len(doc): break
        
        if not valid_win: continue
        
        # Chunk checks
        w_list = doc.lower().split()
        mb, mv, m_mean = get_chunk_metrics(w_list)
        if mb >= thr_cb and mv >= thr_cv and m_mean >= thr_mean:
            s2_candidates.append(idx)
            s2_scores[idx] = m_mean

    # Title dedupe Stage 2
    s2_final = []
    titles_s2 = {}
    non_titled_s2 = []
    for idx in s2_candidates:
        doc = docs[idx]
        first = doc.splitlines()[0] if doc else ""
        if first.startswith('='):
            key = first.strip().replace('=', '').strip()
            if key not in titles_s2: titles_s2[key] = []
            titles_s2[key].append((s2_scores[idx], idx))
        else:
            non_titled_s2.append(idx)
    for key in titles_s2:
        titles_s2[key].sort(key=lambda x: x[0], reverse=True)
        non_titled_s2.append(titles_s2[key][0][1])
    
    # Token budget
    token_data = {idx: len(enc.encode_ordinary(docs[idx])) for idx in unique_indices}
    total_s2 = sum(token_data[i] for i in non_titled_s2)
    
    if total_s2 > T:
        sorted_s2 = sorted(non_titled_s2, key=lambda x: s2_scores.get(x, 0))
        current_total = total_s2
        dropped = set()
        for idx in sorted_s2:
            if current_total <= T: break
            current_total -= token_data[idx]
            dropped.add(idx)
        final_indices = [i for i in non_titled_s2 if i not in dropped]
    elif total_s2 < 0.5 * T:
        final_indices = s_indices
    else:
        final_indices = non_titled_s2

    rng.shuffle(final_indices)
    return final_indices
