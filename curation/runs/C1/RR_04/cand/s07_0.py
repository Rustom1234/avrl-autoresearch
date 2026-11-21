import os
import random
import zlib
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Prep V (alphabetical words in target_sample)
    V = set()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        for w in words:
            V.add(w)
            
    # Scoring system for survivors
    scored_docs = []

    for i, doc in enumerate(docs):
        # 3. Spam and template check
        if len(doc) < 200:
            continue
        
        compressed_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if compressed_ratio < 0.30:
            continue
            
        doc_lower = doc.lower()
        spam_phrases = [
            'click here to order', 'buy now', 'free shipping', 
            'discount code', 'coupon', 'for sale |', 
            'limited offer', 'best price'
        ]
        if any(phrase in doc_lower for phrase in spam_phrases):
            continue

        # 1. Letter-scramble check
        scramble_tokens = re.findall(r'[a-zA-Z]{4,}', doc_lower)
        if len(scramble_tokens) >= 20:
            match_count = sum(1 for w in scramble_tokens if w in V)
            if match_count / len(scramble_tokens) < 0.5:
                continue

        # 2. Word-shuffle check
        tokens = doc.split()
        if len(tokens) > 0:
            # (a) 'the' preceding token check
            pre_the_count = 0
            the_count = 0
            pre_words = {'of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from', 'with', 'as', 'that', 'was', 'is'}
            for j in range(len(tokens)):
                if tokens[j].lower() == 'the':
                    the_count += 1
                    if j > 0 and tokens[j-1].lower() in pre_words:
                        pre_the_count += 1
            if the_count >= 10 and (pre_the_count / the_count) < 0.25:
                continue

            # (b) '.' punctuation check
            dot_count = 0
            dot_followed_lower = 0
            for j in range(len(tokens) - 1):
                t = tokens[j]
                if t == '.' or t.endswith('.'):
                    dot_count += 1
                    next_t = tokens[j+1]
                    if next_t and next_t[0].islower():
                        dot_followed_lower += 1
            if dot_count >= 8 and (dot_followed_lower / dot_count) > 0.4:
                continue
            
            # (c) Adjacent identical function-word pairs
            pair_count = 0
            if len(tokens) >= 4:
                for j in range(len(tokens) - 3):
                    chunk = [t.lower() for t in tokens[j:j+4]]
                    if chunk == ['the', 'of', 'the', 'the']:
                        pair_count += 1
            
            for j in range(len(tokens) - 1):
                pair = (tokens[j].lower(), tokens[j+1].lower())
                if pair in [('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')]:
                    pair_count += 1
            
            if (pair_count / len(tokens)) > 0.005:
                continue

        # 4. Dedupe
        norm_text = " ".join(tokens)
        h1 = hash(norm_text)
        h2 = hash(norm_text[:300])
        # This dedupe check is global-ish in the loop; we use a local set to filter unique candidates
        # to be sorted, then we return indices.
        
    # Since dedupe must be preserved and Wiki score is primary, 
    # we re-run a pass to get unique documents that pass filters.
    unique_docs = []
    seen_hashes = set()
    
    for i, doc in enumerate(docs):
        # Repeat filters (Simplified for the logic flow)
        if len(doc) < 200: continue
        comp = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if comp < 0.30: continue
        dl = doc.lower()
        if any(p in dl for p in ['click here to order', 'buy now', 'free shipping', 'discount code', 'coupon', 'for sale |', 'limited offer', 'best price']): continue
        
        scram = re.findall(r'[a-zA-Z]{4,}', dl)
        if len(scram) >= 20:
            m = sum(1 for w in scram if w in V)
            if m / len(scram) < 0.5: continue
        
        tks = doc.split()
        if len(tks) > 0:
            tc = 0
            ptc = 0
            pre_ws = {'of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from', 'with', 'as', 'that', 'was', 'is'}
            for j in range(len(tks)):
                if tks[j].lower() == 'the':
                    tc += 1
                    if j > 0 and tks[j-1].lower() in pre_ws: ptc += 1
            if tc >= 10 and (ptc / tc) < 0.25: continue
            
            dc = 0
            dfl = 0
            for j in range(len(tks) - 1):
                if tks[j] == '.' or tks[j].endswith('.'):
                    dc += 1
                    if tks[j+1] and tks[j+1][0].islower(): dfl += 1
            if dc >= 8 and (dfl / dc) > 0.4: continue
            
            pc = 0
            if len(tks) >= 4:
                for j in range(len(tks) - 3):
                    if [x.lower() for x in tks[j:j+4]] == ['the', 'of', 'the', 'the']: pc += 1
            for j in range(len(tks) - 1):
                p = (tks[j].lower(), tks[j+1].lower())
                if p in [('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')]: pc += 1
            if (pc / len(tks)) > 0.005: continue
            
        nh = hash(norm_text := " ".join(tks))
        nh2 = hash(norm_text[:300])
        if nh not in seen_hashes:
            seen_hashes.add(nh)
            # Calculate Wiki score
            markers = doc.count('= ') + doc.count(' , ') + doc.count(' . ') + doc.count(' @-@ ')
            score = markers / len(doc)
            unique_docs.append((score, i, len(doc)))

    # Sort by score descending, then original index
    unique_docs.sort(key=lambda x: (-x[0], x[1]))
    
    kept_indices = []
    current_tokens = 0
    
    # The prompt implies we select until we reach T, but if we run out, we cycle.
    # "Select the top documents until the cumulative token count reaches T"
    # To allow cycling, we pick the best ones first.
    for score, idx, length in unique_docs:
        if current_tokens < T:
            kept_indices.append(idx)
            current_tokens += length
            
    # If we haven't reached T, the prompt says "allow the trainer to cycle".
    # Since the trainer cycles automatically if the list is shorter than T, 
    # we just return the ordered list of best documents.
    
    # Because the requirement says "returns an ordered list of indices", 
    # and the trainer handles the cycling, we return the best ones found.
    return kept_indices
