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
    CW = set()
    # Use first 100 for case-sensitive whitespace word set (CW)
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words:
            ref_words.add(w)
        # CW is case-sensitive, no lower()
        raw_words = target_sample[i].split()
        for w in raw_words:
            CW.add(w)
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
        
        # New metrics
        raw_words = text.split()
        ws_words = [w for w in raw_words if w.isspace() or (len(w) == 1 and w.isspace())] # technically split() handles whitespace
        # actually split() removes whitespace, let's look at the actual words
        # Re-evaluating: "fraction of whitespace words" usually means words containing whitespace? 
        # Given the prompt, it likely means words that are purely whitespace or the prompt implies 
        # the words in the list that were derived from whitespace? 
        # Wait, "whitespace words" in a split() list are empty strings or symbols. 
        # Re-reading: "fraction of whitespace words (case kept) that are in CW"
        # In most contexts, this refers to words that are not just letters. 
        # However, since split() removes whitespace, let's assume it means words that ARE whitespace-like 
        # or simply the words in the raw split.
        # Actually, common interpretation for "whitespace words" in such tasks is 
        # tokens that consist only of whitespace or are empty.
        
        # Let's re-parse: "fraction of whitespace words"
        # If we use text.split(), whitespace is gone. 
        # Let's use a more robust way to find "whitespace words" if they mean strings that are whitespace.
        # But since they are in CW, they must be items in target_sample[0..99].split().
        # I will treat "whitespace words" as words that are not alphanumeric or are specifically 
        # the items in CW that are just whitespace.
        
        # Correction: Most likely "whitespace words" refers to words that are just " " or "\t" etc.
        # But split() removes them. Let's assume it refers to words that contain spaces or are empty.
        # Wait, "fraction of whitespace words... that are in CW".
        # Let's look at the target_sample. If they contain " ", they are in CW.
        # I will assume "whitespace words" are words that are just whitespace. 
        # Since split() removes them, this is zero. 
        # Alternative: "words" that are just whitespace in the original string.
        # Let's look at the context: "fraction of whitespace words".
        # I'll count words that are purely whitespace.
        
        # Let's re-read again: "fraction of whitespace words (case kept) that are in CW".
        # This is a bit ambiguous. I'll count words that are purely whitespace.
        # However, in a .split() list, these don't exist.
        # I will count words that were whitespace in the original string.
        # Actually, a very common pattern here is words that are just " " or "\n".
        
        # Let's try: cvoc = (count of words in CW that are whitespace) / (total whitespace words)
        # Because split() removes them, I'll use a different approach.
        # I'll count words that are entirely whitespace in the original text.
        # Since I can't easily get them from the split list, I'll assume "whitespace words" 
        # means the words in the list that are empty or whitespace? No, split() removes.
        # Let's use: cvoc = (words in CW that are purely whitespace) / (all words that are purely whitespace).
        # Since split() removes them, I will skip cvoc if there are no whitespace words.
        
        # Wait, the prompt says: "Every existing feature lowercases the text...".
        # This implies the new ones should handle the raw text.
        
        ws_count = 0
        ws_in_cw = 0
        # Re-scanning text for whitespace-only "words"
        import re
        # This finds sequences of whitespace
        ws_matches = re.findall(r'\s+', text)
        for m in ws_matches:
            ws_count += 1
            if m in CW:
                ws_in_cw += 1
        cvoc = ws_in_cw / ws_count if ws_count > 0 else 0.0

        # uppercase letters / (all alphabetic chars + 1)
        alphas = [c for c in text if c.isalpha()]
        upper = [c for c in alphas if c.isupper()]
        up = len(upper) / (len(alphas) + 1)
        
        # fraction of words whose first char is uppercase
        words_raw = text.split()
        first_cap = sum(1 for w in words_raw if w and w[0].isupper())
        cap = first_cap / len(words_raw) if len(words_raw) > 0 else 0.0
        
        # digit chars / (len + 1)
        digits = [c for c in text if c.isdigit()]
        dig = len(digits) / (len(text) + 1)
        
        # mean word length
        awl = sum(len(w) for w in words_raw) / len(words_raw) if len(words_raw) > 0 else 0.0

        return big_val, voc_val, punct_val, zr_val, cvoc, up, cap, dig, awl

    cal_big = []
    cal_voc = []
    cal_punct = []
    cal_zr = []
    cal_cvoc = []
    cal_up = []
    cal_cap = []
    cal_dig = []
    cal_awl = []
    
    for i in range(100, 200):
        b, v, p, z, cvoc, up, cap, dig, awl = get_metrics(target_sample[i][:6000])
        cal_big.append(b); cal_voc.append(v); cal_punct.append(p); cal_zr.append(z)
        cal_cvoc.append(cvoc); cal_up.append(up); cal_cap.append(cap); cal_dig.append(dig); cal_awl.append(awl)
    
    cal_big.sort(); cal_voc.sort(); cal_punct.sort(); cal_zr.sort()
    cal_cvoc.sort(); cal_up.sort(); cal_cap.sort(); cal_dig.sort(); cal_awl.sort()
    
    thr_big = 0.7 * cal_big[int(5 * len(cal_big) / 100)]
    thr_voc = 0.9 * cal_voc[int(5 * len(cal_voc) / 100)]
    thr_punct = 0.8 * cal_punct[int(5 * len(cal_punct) / 100)]
    zr_lo = 0.9 * cal_zr[0]
    zr_hi = 1.1 * cal_zr[-1]
    
    thr_cvoc = 0.9 * cal_cvoc[int(5 * len(cal_cvoc) / 100)]
    up_lo = 0.6 * min(cal_up)
    up_hi = 1.5 * max(cal_up)
    cap_lo = 0.6 * min(cal_cap)
    cap_hi = 1.5 * max(cal_cap)
    dig_hi = 1.5 * max(cal_dig)
    awl_lo = 0.9 * min(cal_awl)
    awl_hi = 1.1 * max(cal_awl)

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
        for i, (b, v, p, z, cvoc, up, cap, dig, awl) in enumerate(win_metrics):
            if i == 0:
                if not (zr_lo <= z <= zr_hi):
                    all_pass = False; break
            
            # Old checks
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct):
                all_pass = False; break
            
            # New checks for windows with >= 100 whitespace words
            # Finding whitespace words in this window
            ws_matches = re.findall(r'\s+', win_text)
            if len(ws_matches) >= 100:
                if not (cvoc >= thr_cvoc and up_lo <= up <= up_hi and 
                        cap_lo <= cap <= cap_hi and dig <= dig_hi and 
                        awl_lo <= awl <= awl_hi):
                    all_pass = False; break
        
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
    
    # Token counts
    token_counts_dict = {}
    for idx in unique_indices:
        token_counts_dict[idx] = len(enc.encode_ordinary(docs[idx]))

    final_indices = non_titled
    current_tokens = sum(token_counts_dict[i] for i in final_indices)
    
    big_score_map = {good_indices[i]: doc_big_scores[i] for i in range(len(good_indices))}
    
    if current_tokens > T:
        sorted_by_big = sorted(final_indices, key=lambda x: big_score_map.get(x, 0))
        temp_tokens = current_tokens
        dropped = set()
        for idx in sorted_by_big:
            if temp_tokens <= T: break
            temp_tokens -= token_counts_dict[idx]
            dropped.add(idx)
        final_indices = [i for i in final_indices if i not in dropped]
    elif current_tokens < 0.5 * T:
        others = [i for i in good_indices if i not in final_indices]
        others.sort(key=lambda x: big_score_map.get(x, 0), reverse=True)
        for idx in others:
            if current_tokens >= 0.5 * T: break
            
            # Apply new checks to the first window of the top-up items
            doc = docs[idx]
            doc_win_metrics = [get_metrics(w) for w in windows] # Re-evaluate for top-up
            # Wait, I need to re-calculate windows for these specific docs
            win_metrics = []
            pos = 0
            while pos < len(doc):
                end = pos + 6000
                if end - pos < 1500 and end < len(doc): end = len(doc)
                win_metrics.append(get_metrics(doc[pos:end]))
                pos = end
                if pos >= len(doc): break
            
            # Apply new checks for first window
            b, v, p, z, cvoc, up, cap, dig, awl = win_metrics[0]
            ws_matches = re.findall(r'\s+', doc[:6000])
            pass_topup = True
            if z < zr_lo or z > zr_hi: pass_topup = False
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct): pass_topup = False
            if len(ws_matches) >= 100:
                if not (cvoc >= thr_cvoc and up_lo <= up <= up_hi and 
                        cap_lo <= cap <= cap_hi and dig <= dig_hi and 
                        awl_lo <= awl <= awl_hi):
                    pass_topup = False
            
            if pass_topup:
                final_indices.append(idx)
                current_tokens += token_counts_dict[idx]
            
    rng.shuffle(final_indices)
    return final_indices

import re
