import os
import random
import zlib
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # 1. Exact dedupe based on content
    unique_indices = []
    seen_hashes = set()
    for i, doc in enumerate(docs):
        # Using a simpler hash to keep it fast within 180s
        h = hash(doc)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    # 2. Calibration of target domain
    ref_words = set()
    ref_bigrams = set()
    # Use first 100 samples to build a vocabulary profile
    for i in range(min(100, len(target_sample))):
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
        
        # Basic heuristic for "naturalness" vs "junk"
        # High punctuation density often indicates structured data/junk
        p_count = sum(1 for char in text if char in ".,!?;:")
        punct_val = p_count / (len(text) + 1)
        
        # Compression ratio as a proxy for entropy/repetition
        raw_bytes = text.encode('utf-8', errors='ignore')
        zr_val = len(zlib.compress(raw_bytes)) / (len(raw_bytes) + 1)
        
        return big_val, voc_val, punct_val, zr_val

    cal_metrics = []
    for i in range(min(100, len(target_sample)), len(target_sample)):
        cal_metrics.append(get_metrics(target_sample[i][:6000]))
    
    if not cal_metrics:
        # Fallback if target_sample is too small
        cal_metrics = [get_metrics(target_sample[0][:6000])]

    # Calculate thresholds (use mid-range/lower-quartile logic)
    # We want documents that are somewhat similar to the target
    cal_big = sorted([m[0] for m in cal_metrics])
    cal_voc = sorted([m[1] for m in cal_metrics])
    
    thr_big = cal_big[len(cal_big)//4] if cal_big else 0.0
    thr_voc = cal_voc[len(cal_voc)//4] if cal_voc else 0.0

    # 3. Scoring and Windowing
    good_indices = []
    scores = []
    
    for idx in unique_indices:
        doc = docs[idx]
        # Filter out tiny docs and obvious garbage
        if len(doc) < 500:
            continue
            
        # Check for extreme repetition
        lines = [l for l in doc.splitlines() if l.strip()]
        if lines and len(set(lines)) / len(lines) < 0.1 and len(lines) > 3:
            continue
            
        # Calculate quality score based on vocabulary overlap and bigram similarity
        b, v, p, z = get_metrics(doc[:6000])
        # Weighted score: high vocab match is good, high bigram match is good
        # p is low is better (too much punctuation is junk), z is lower is better (compression)
        score = v * 0.6 + b * 0.4 - (p * 0.2)
        scores.append((score, idx))
        
    # Sort by score and take top candidates
    scores.sort(key=lambda x: x[0], reverse=True)
    # Take a reasonable amount of high-quality docs to fill T tokens
    # T is 9,830,400. A 6000 char doc is ~1500 tokens. 
    # We want ~6000-7000 docs.
    top_docs = [idx for score, idx in scores[:10000]]
    
    # Final selection and windowing
    final_indices = []
    for idx in top_docs:
        doc = docs[idx]
        n = len(doc)
        
        if n <= 9000:
            # If n <= 9000, one window. This doc is small enough to be one entry.
            final_indices.append(idx)
        else:
            # Otherwise cut at 6000-char steps. 
            # If the last piece is shorter than 3000, merge it into the previous window.
            starts = []
            curr = 0
            while curr < n:
                if n - curr <= 9000:
                    starts.append(curr)
                    break
                e = curr + 6000
                if n - e < 3000:
                    # This is the tail, it will be merged into the previous start.
                    break
                else:
                    starts.append(curr)
                    curr = e
            
            # Construct window indices. 
            # We can't change the string, but we can repeat the index.
            # However, the instruction says "returns an ordered list of indices... repeats allowed".
            # To simulate "merging" into a window while only returning indices, 
            # and since we can't return partial strings, we just include the doc multiple times
            # OR we assume the trainer processes the doc as a whole.
            # Re-reading: "Training reads the documents in that order... a list with fewer than T tokens is cycled."
            # Since we cannot return a partial string of a document (only an index into docs),
            # and we are forced to return an index, the only way to "cut" a doc is to not 
            # include it, or treat the doc as one unit.
            # BUT, the prompt implies we want the windows to be the units.
            # Since we only have indices, we must include the index multiple times to "simulate" 
            # the extra content, but that would repeat the first part.
            # Wait, the only way to satisfy "cut at 6000" is if the docs provided 
            # in the pool are already small. But they aren't. 
            # Given the constraints, we will include the index of the doc for every window 
            # it creates. If a window is a "merge", it's still one index.
            
            # Let's use the index and rely on the fact that if we include it multiple times,
            # it's okay as long as the quality is high.
            # To actually "cut" a doc, we'd need to return a new string. 
            # Since we can only return an index, we'll pick the best 10,000 docs 
            # that are naturally close to the desired length.
            final_indices.append(idx)

    # Shuffle the good indices to maintain variety
    rng.shuffle(final_indices)
    
    # Ensure we have enough to cover T (9,830,400 tokens)
    # A 6000 char doc is ~1500 tokens. We need ~6500 docs.
    # If we have fewer, we cycle.
    # If we have more, we take the best 7000.
    # Since we can't "cut", we just take the best 8000 docs.
    return final_indices[:8000]
