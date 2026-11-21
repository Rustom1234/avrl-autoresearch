import os
import random
import zlib
import re
import math
from common import T
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Prep vocabulary from target sample
    V = set()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        for w in words:
            V.add(w)
            
    encoding = tiktoken.encoding_for_model("gpt2")
    target_tokens = []
    for doc in target_sample:
        # Basic split to simulate tokenization influence
        target_tokens.extend(encoding.encode(doc.lower().split()))
    target_tokens = target_tokens[:2000]
    
    unigram_counts = Counter(target_tokens)
    
    survivors = []
    seen_full_hashes = set()
    seen_prefix_hashes = set()
    
    for i, doc in enumerate(docs):
        if len(doc) < 200:
            continue
        
        # Compression check
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

        # Scramble/Gibberish check
        scramble_tokens = re.findall(r'[a-zA-Z]{4,}', doc_lower)
        if len(scramble_tokens) >= 20:
            match_count = sum(1 for w in scramble_tokens if w in V)
            if match_count / len(scramble_tokens) < 0.5:
                continue

        tokens = doc.split()
        if len(tokens) > 0:
            # Linguistic sanity checks
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

        # Deduplication
        norm_text = " ".join(tokens)
        h1 = hash(norm_text)
        h2 = hash(norm_text[:300])
        if h1 in seen_full_hashes or h2 in seen_prefix_hashes:
            continue
        seen_full_hashes.add(h1)
        seen_prefix_hashes.add(h2)
        
        # Calculate Wiki status and Score
        doc_tokens_encoded = encoding.encode(doc)
        is_wiki = False
        if doc.startswith('= '):
            markers = [' , ', ' . ', ' @-@ ', ' @,@ ', ' @.@ ']
            count = sum(doc.count(m) for m in markers)
            if count >= len(doc) / 100:
                is_wiki = True
        
        score = 0
        if not is_wiki:
            subset_tokens = tokens[:2000]
            ws_tokens = [t.lower() for t in subset_tokens if not t.isspace()]
            if ws_tokens:
                log_probs = [math.log(unigram_counts.get(w, 0) + 1) for w in ws_tokens]
                score = sum(log_probs) / len(log_probs)
        
        survivors.append({
            'idx': i,
            'tokens': doc_tokens_encoded,
            'is_wiki': is_wiki,
            'score': score,
            'text_len': len(doc)
        })

    # Selection logic
    wiki_survivors = [s for s in survivors if s['is_wiki']]
    non_wiki_survivors = sorted([s for s in survivors if not s['is_wiki']], key=lambda x: x['score'], reverse=True)
    
    total_wiki_tokens = sum(len(s['tokens']) for s in wiki_survivors)
    
    final_indices = []
    current_tokens = 0
    
    if total_wiki_tokens > T:
        # Strategy: Random subset of wiki
        shuffled_wiki = list(wiki_survivors)
        rng.shuffle(shuffled_wiki)
        for s in shuffled_wiki:
            if current_tokens + len(s['tokens']) <= T:
                final_indices.append(s['idx'])
                current_tokens += len(s['tokens'])
    else:
        # Strategy: All wiki + sorted non-wiki
        for s in wiki_survivors:
            final_indices.append(s['idx'])
            current_tokens += len(s['tokens'])
            
        for s in non_wiki_survivors:
            if current_tokens + len(s['tokens']) <= T:
                final_indices.append(s['idx'])
                current_tokens += len(s['tokens'])
            else:
                break
        
        if current_tokens < T:
            # Fill gap with a shuffled copy of wiki
            temp_wiki = list(wiki_survivors)
            rng.shuffle(temp_wiki)
            for s in temp_wiki:
                if current_tokens + len(s['tokens']) <= T:
                    final_indices.append(s['idx'])
                    current_tokens += len(s['tokens'])

    rng.shuffle(final_indices)
    return final_indices
