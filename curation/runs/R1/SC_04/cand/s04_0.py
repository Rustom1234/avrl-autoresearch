import os
import random
import zlib
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Exact dedupe
    unique_docs = []
    seen_hashes = set()
    for doc in docs:
        h = hash(" ".join(doc.split()))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    
    def get_features(text, bigram_set, word_set):
        words = text.lower().split()
        # (a) big
        big_count = 0
        total_bigrams = 0
        for i in range(len(words) - 1):
            b = f"{words[i]} {words[i+1]}"
            if b in bigram_set:
                big_count += 1
            total_bigrams += 1
        big_val = big_count / total_bigrams if total_bigrams > 0 else 0.0
        
        # (b) voc
        voc_count = sum(1 for w in words if w in word_set)
        voc_val = voc_count / len(words) if len(words) > 0 else 0.0
        
        # (c) punct
        # Note: use the raw text for punctuation as per instruction
        p1 = text.count(' , ')
        p2 = text.count(' . ')
        p3 = text.count(',')
        p4 = text.count('.')
        punct_val = (p1 + p2) / (p3 + p4 + 1)
        
        # (d) zr
        raw_bytes = text.encode('utf-8', errors='ignore')
        zr_val = len(zlib.compress(raw_bytes)) / len(raw_bytes) if len(raw_bytes) > 0 else 0.0
        
        # (e) nw
        nw_val = len(words)
        
        # (f) rep
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) > 0:
            unique_lines = len(set(lines))
            rep_val = 1 - (unique_lines / len(lines))
        else:
            rep_val = 0.0
            
        return big_val, voc_val, punct_val, zr_val, nw_val, rep_val

    # Build reference sets from target_sample[0..99]
    ref_words = set()
    ref_bigrams = set()
    for i in range(100):
        ts = target_sample[i].lower().split()
        for w in ts:
            ref_words.add(w)
        for j in range(len(ts) - 1):
            ref_bigrams.add(f"{ts[j]} {ts[j+1]}")

    # Calibration features for target_sample[100..199]
    cal_big = []
    cal_voc = []
    cal_punct = []
    cal_zr = []
    for i in range(100, 200):
        b, v, p, z, _, _ = get_features(target_sample[i], ref_bigrams, ref_words)
        cal_big.append(b)
        cal_voc.append(v)
        cal_punct.append(p)
        cal_zr.append(z)

    cal_big.sort()
    cal_voc.sort()
    cal_punct.sort()
    cal_zr.sort()

    thr_big = 0.7 * cal_big[int(5 * (len(cal_big)/100))]
    thr_voc = 0.9 * cal_voc[int(5 * (len(cal_voc)/100))]
    thr_punct = 0.8 * cal_punct[int(5 * (len(cal_punct)/100))]
    zr_lo = 0.9 * cal_zr[0]
    zr_hi = 1.1 * cal_zr[-1]

    # Process all docs
    good_indices = []
    doc_features = [] # list of (big, voc, punct, zr, nw, rep, original_idx)
    
    for i, doc in enumerate(unique_docs):
        # Use first 6000 chars for features as per instruction
        feat_text = doc[:6000]
        b, v, p, z, nw, rep = get_features(feat_text, ref_bigrams, ref_words)
        
        is_good = (b >= thr_big) and (v >= thr_voc) and (p >= thr_punct) and \
                   (z >= zr_lo) and (z <= zr_hi) and (nw >= 40) and (rep <= 0.3)
        
        if is_good:
            good_indices.append(i)
            doc_features.append((b, v, p, z, nw, rep, i))

    # 5. Title dedupe
    titles = {} # map stripped first line to list of (big, original_idx)
    final_good_indices = []
    
    # Group by title for dedupe
    temp_good = []
    for i in good_indices:
        doc = unique_docs[i]
        lines = doc.splitlines()
        first = lines[0] if lines else ""
        if first.startswith('='):
            stripped = first.strip().replace('=', '').strip()
            if stripped not in titles: titles[stripped] = []
            titles[stripped].append((doc_features[good_indices.index(i)][0], i))
        else:
            temp_good.append(good_indices[good_indices.index(i)]) # wait, this is slightly buggy in logic
            # Let's just do it sequentially to keep it simple
            pass
            
    # Corrected Title Dedupe Logic
    titles = {}
    kept_indices = []
    for i in good_indices:
        doc = unique_docs[i]
        lines = doc.splitlines()
        first = lines[0] if lines else ""
        if first.startswith('='):
            stripped = first.strip().replace('=', '').strip()
            if stripped not in titles:
                titles[stripped] = []
            titles[stripped].append((doc_features[good_indices.index(i)][0], i))
        else:
            kept_indices.append(i)
    
    # Pick best big for titles
    for stripped in titles:
        titles[stripped].sort(key=lambda x: x[0], reverse=True)
        kept_indices.append(titles[stripped][0][1])
    
    # Re-map features to kept_indices
    # We need to find the big score for each kept index
    kept_data = []
    for idx in kept_indices:
        # find the original good_indices position
        pos = good_indices.index(idx)
        kept_data.append(doc_features[pos])

    # 6. Token counting and final filtering
    enc = tiktoken.get_encoding("gpt2")
    total_tokens = 0
    for _, _, _, _, nw, _, _ in kept_data:
        # We use the full doc for token count, but nw is from first 6000 chars? 
        # The instruction says "Count tokens of the good docs".
        # It doesn't specify 6000 chars for count, so we use full doc.
        # However, we need to know which doc is which.
        pass

    # Let's rebuild kept_data to include the actual doc to get tokens
    # but we must maintain the order for the final shuffle.
    final_docs = [unique_docs[idx] for idx in kept_indices]
    token_counts = [len(enc.encode_ordinary(d)) for d in final_docs]
    
    # Re-calculate big scores for the final_docs to handle the drop/add logic
    # We'll use the 6000-char versions
    final_big_scores = []
    for d in final_docs:
        b, _, _, _, _, _ = get_features(d[:6000], ref_bigrams, ref_words)
        final_big_scores.append(b)

    current_total = sum(token_counts)
    
    if current_total > T:
        # Drop lowest big until about T
        # Sort by big score
        indices = sorted(range(len(final_docs)), key=lambda k: final_big_scores[k])
        for i in indices:
            if current_total <= T:
                break
            current_total -= token_counts[i]
            # This is tricky because we need to actually remove them
            # Let's just filter the list
            pass
        # Redo this:
        sorted_indices = sorted(range(len(final_docs)), key=lambda k: final_big_scores[k])
        to_remove = set()
        temp_sum = current_total
        for i in sorted_indices:
            if temp_sum <= T:
                break
            temp_sum -= token_counts[i]
            to_remove.add(i)
        
        final_final_indices = [i for i in range(len(final_docs)) if i not in to_remove]
    elif current_total < 0.5 * T:
        # Add deduped docs with nw >= 40 in descending big until 0.5 T
        # "deduped docs" refers to unique_docs that were not 'good'
        # or just any unique_docs? The prompt implies "deduped docs" (from step 1)
        # that weren't included.
        # Let's find docs that were skipped (nw >= 40)
        skipped = []
        all_indices = set(range(len(unique_docs)))
        included_indices = set(kept_indices)
        for i in all_indices:
            if i not in included_indices:
                _, _, _, _, nw, _ = get_features(unique_docs[i][:6000], ref_bigrams, ref_words)
                if nw >= 40:
                    b, _, _, _, _, _ = get_features(unique_docs[i][:6000], ref_bigrams, ref_words)
                    skipped.append((b, i))
        
        skipped.sort(key=lambda x: x[0], reverse=True)
        
        # Title deduped again? "add deduped docs" might mean the result of step 1
        # or step 5. Let's assume result of step 1 (unique_docs).
        # If we need to keep it unique, we handle that here.
        already_included_idx = set(kept_indices)
        for b, i in skipped:
            if i not in already_included_idx:
                final_final_indices = kept_indices + [i] # Placeholder logic
                # Wait, need to build final_final_indices correctly
                pass
        # Let's simplify:
        # 1. Identify all unique docs with nw >= 40 and big score
        # 2. Remove those already in kept_indices
        # 3. Sort by big score desc
        # 4. Add until 0.5 T
        
        skipped_pool = []
        for i, doc in enumerate(unique_docs):
            if i not in included_indices:
                _, _, _, _, nw, _ = get_features(doc[:6000], ref_bigrams, ref_words)
                if nw >= 40:
                    b, _, _, _, _, _ = get_features(doc[:6000], ref_bigrams, ref_words)
                    skipped_pool.append((b, i))
        skipped_pool.sort(key=lambda x: x[0], reverse=True)
        
        final_final_indices = kept_indices[:]
        current_total = sum(token_counts)
        for b, i in skipped_pool:
            if current_total >= 0.5 * T:
                break
            final_final_indices.append(i)
            current_total += len(enc.encode_ordinary(unique_docs[i]))
    else:
        final_final_indices = kept_indices
        
    # Final map to original indices of unique_docs
    # unique_docs = docs[indices] -> docs[unique_docs[idx]]
    # actually kept_indices are indices into unique_docs.
    # The return must be indices into docs.
    
    # To get indices into docs, we need to track unique_docs indices
    # unique_indices = [i for i in range(len(docs)) if i in seen_hashes] # No, this is wrong.
    # Let's restart the indexing logic clearly.
    
    # RE-STARTING LOGIC FOR INDEX TRACKING
    # 1. Unique docs
    unique_indices = []
    seen_hashes = set()
    for i in range(len(docs)):
        h = hash(" ".join(docs[i].split()))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
            
    # 2. Features (re-run using unique_indices)
    # ... (This is why the previous flow was confusing)
    # Let's use a list of (original_idx, doc_string)
    
    # Let's just fix the current code by mapping back
    # current kept_indices are indices into unique_docs
    # So final_indices_into_docs = [unique_indices[i] for i in final_final_indices]
    
    # Re-calculating to be sure:
    # The goal is to return indices into `docs`.
    # unique_indices[j] is the index into `docs`.
    
    # I will rewrite the selection slightly to keep track of `docs` indices directly.
    
    return [unique_indices[i] for i in final_final_indices]

# Since I must provide the complete file, I'll rewrite the logic cleanly to ensure
# indices map correctly from the start.
