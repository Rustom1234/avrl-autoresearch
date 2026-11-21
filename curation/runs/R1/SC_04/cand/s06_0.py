import os
import random
import zlib
import tiktoken
from common import T
import numpy as np

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Exact dedupe
    unique_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        h = zlib.crc32(doc.replace(" ", "").encode('utf-8'))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    # Helper to get window metrics
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

    # Calibration
    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        words = target_sample[i].lower().split()
        for w in words:
            ref_words.add(w)
        for j in range(len(words) - 1):
            ref_bigrams.add(f"{words[j]} {words[j+1]}")

    def get_windows(text):
        wins = []
        pos = 0
        while pos < len(text):
            end = pos + 1500
            if end < len(text):
                # Find nearest whitespace
                idx = text.rfind(' ', end - 1, end)
                if idx > pos:
                    end = idx + 1
                else:
                    end = pos + 1
            else:
                end = len(text)
            
            win = text[pos:end]
            if win:
                wins.append(win)
            pos = end
        
        # Merge small tail
        if len(wins) > 1 and len(wins[-1]) < 600:
            wins[-2] += wins[-1]
            wins.pop()
        return wins

    cal_big, cal_voc, cal_punct = [], [], []
    for i in range(100, 200):
        wins = get_windows(target_sample[i])
        for w in wins:
            b, v, p, z = get_metrics(w)
            cal_big.append(b); cal_voc.append(v); cal_punct.append(p)
    
    thr_big = 0.75 * np.percentile(cal_big, 2)
    thr_voc = 0.92 * np.percentile(cal_voc, 2)
    thr_punct = 0.70 * np.percentile(cal_punct, 2)

    # Target first window ZR bounds (at 6000)
    zr_vals = []
    for i in range(100, 200):
        txt = target_sample[i][:6000]
        _, _, _, z = get_metrics(txt)
        zr_vals.append(z)
    zr_lo = 0.9 * min(zr_vals)
    zr_hi = 1.1 * max(zr_vals)

    # Pool filtering
    good_indices = []
    doc_big_scores = []
    
    # Pre-calculate all token counts
    doc_tokens = {i: len(enc.encode_ordinary(docs[i])) for i in unique_indices}

    for idx in unique_indices:
        doc = docs[idx]
        words_all = doc.split()
        lines = [l for l in doc.splitlines() if l.strip()]
        rep_val = 1 - (len(set(lines)) / len(lines)) if lines else 0
        
        if len(words_all) < 40 or rep_val > 0.3:
            continue
            
        wins = get_windows(doc)
        win_metrics = [get_metrics(w) for w in wins]
        
        # Check ZR on first 6000 chars
        first_6k_z = get_metrics(doc[:6000])[3]
        if not (zr_lo <= first_6k_z <= zr_hi):
            continue

        # Logic for every window meeting thresholds
        all_pass = True
        fail_count = 0
        for i, (b, v, p, z) in enumerate(win_metrics):
            if not (b >= thr_big and v >= thr_voc and p >= thr_punct):
                # Tolerance for 1 odd section
                if b >= 0.6 * thr_big and v >= 0.9 * thr_voc:
                    fail_count += 1
                else:
                    all_pass = False
                    break
        
        if not all_pass or (fail_count > 1 and len(wins) >= 8):
            continue
        if not all_pass and len(wins) >= 8 and fail_count == 1:
            pass # Allowed
        else:
            if not all_pass: continue

        good_indices.append(idx)
        doc_big_scores.append(sum(m[0] for m in win_metrics) / len(win_metrics))

    # Title Dedupe
    titles = {}
    non_titled = []
    title_map_best_score = {}
    
    for i, idx in enumerate(good_indices):
        doc = docs[idx]
        first_line = doc.splitlines()[0] if doc else ""
        if first_line.startswith('='):
            key = first_line.strip().replace('=', '').strip()
            if key not in titles: titles[key] = []
            titles[key].append((doc_big_scores[i], idx))
        else:
            non_titled.append(idx)
            
    for key in titles:
        titles[key].sort(key=lambda x: x[0], reverse=True)
        best_idx = titles[key][0][1]
        non_titled.append(best_idx)
        title_map_best_score[best_idx] = titles[key][0][0]
    
    # Token Budget
    final_indices = non_titled
    current_tokens = sum(doc_tokens[i] for i in final_indices)
    
    if current_tokens > T:
        # Sort by mean big score (descending) to keep best
        sorted_final = sorted(final_indices, key=lambda x: title_map_best_score.get(x, 0), reverse=True)
        temp_tokens = current_tokens
        keep = []
        # This logic was slightly different in original: "sort by big score asc to drop lowest"
        # Re-implementing: sort ascending, then keep the head of the list until sum <= T
        sorted_asc = sorted(final_indices, key=lambda x: title_map_best_score.get(x, 0))
        final_indices = []
        temp_tokens = current_tokens
        for idx in sorted_asc:
            if temp_tokens <= T:
                final_indices.append(idx)
            else:
                temp_tokens -= doc_tokens[idx]
        # Actually original code: sorted by score asc, drop lowest (which are at the start of asc)
        # Let's match "Sort by big score asc to drop lowest":
        sorted_by_big = sorted(final_indices, key=lambda x: title_map_best_score.get(x, 0))
        # Wait, the original code was: sorted_by_big = sorted(final_indices, key=lambda x: big_score_map.get(x, 0))
        # Then it dropped from the start (the lowest).
        # To be safe, let's replicate the exact logic:
        sorted_by_big = sorted(final_indices, key=lambda x: title_map_best_score.get(x, 0))
        final_indices = []
        temp_tokens = current_tokens
        for idx in sorted_by_big:
            if temp_tokens <= T:
                final_indices.append(idx)
            else:
                temp_tokens -= doc_tokens[idx]
        # Re-checking: "drop lowest mean big if over T" means the smallest scores go first.
        # The loop above drops the smallest scores first.
        
    # Wait, the prompt says "drop lowest mean big if over T". 
    # My loop above drops smallest scores first. That's correct.
    # Let's re-verify the "drop lowest" logic.
    # If current > T, we need to remove elements until total <= T. 
    # To keep the best, we remove the smallest.
    # So, sort ascending, then skip the head until sum is low enough.
    
    # Let's re-calculate to be certain:
    sorted_by_big = sorted(final_indices, key=lambda x: title_map_best_score.get(x, 0))
    # The first elements in sorted_by_big are the smallest scores.
    # We want to remove those.
    
    # Revised Token Budget to exactly match logic:
    final_indices = non_titled
    current_tokens = sum(doc_tokens[i] for i in final_indices)
    if current_tokens > T:
        sorted_by_big = sorted(final_indices, key=lambda x: title_map_best_score.get(x, 0))
        dropped_indices = set()
        temp_tokens = current_tokens
        for idx in sorted_by_big:
            if temp_tokens <= T: break
            temp_tokens -= doc_tokens[idx]
            dropped_indices.add(idx)
        final_indices = [i for i in final_indices if i not in dropped_indices]
    elif current_tokens < 0.5 * T:
        # Top up from docs that pass first-window test
        others = [i for i in good_indices if i not in final_indices]
        others.sort(key=lambda x: doc_big_scores[good_indices.index(x)], reverse=True)
        for idx in others:
            if current_tokens >= 0.5 * T: break
            final_indices.append(idx)
            current_tokens += doc_tokens[idx]

    rng.shuffle(final_indices)
    return final_indices
