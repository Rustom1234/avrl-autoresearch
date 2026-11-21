import os
import random
import zlib
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    T = 9830400
    target_vocab = set()
    target_bigrams = set()
    
    # Pre-process target sample
    target_words = []
    for s in target_sample:
        words = re.findall(r'[a-zA-Z]+', s.lower())
        target_words.extend(words)
        target_vocab.update(words)
        for i in range(len(words) - 1):
            target_bigrams.add((words[i], words[i+1]))
            
    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # 1. Feature Extraction & 2. Dedupe
    # We use a dictionary to store doc data to help with deduplication and stats
    doc_data = []
    seen_full = set()
    seen_partial = set()
    
    # Global stats for Wiki median/MAD
    wiki_rare_word_rates = []
    wiki_bigram_rates = []

    # First pass for stats and deduplication
    for i, doc in enumerate(docs):
        # Whitespace normalization for hash
        norm_text = " ".join(doc.split())
        f_hash = hash(norm_text)
        p_hash = hash(norm_text[:300])
        
        if f_hash in seen_full or p_hash in seen_partial:
            continue
        seen_full.add(f_hash)
        seen_partial.add(p_hash)

        # Features
        chars = list(doc)
        num_chars = len(doc)
        is_wiki_start = doc.startswith('= ')
        wiki_markers = [' , ', ' . ', ' @-@ ', ' @,@ ', ' ( ']
        marker_count = sum(doc.count(m) for m in wiki_markers)
        is_wiki_marker_density = (marker_count / num_chars) >= 0.01 if num_chars > 0 else False
        is_wiki = is_wiki_start and is_wiki_marker_density
        
        # Rare word rate
        words = re.findall(r'[a-z]+', doc.lower())
        alpha_words = [w for w in words if w.isalpha()]
        
        # We need pool document frequency
        # This is expensive, so we approximate doc frequency by checking 
        # if word appears in very few docs. For the sake of this constraint, 
        # we'll use a simple count in the first 1000 words of the pool as a proxy 
        # or just assume a doc-freq <= 2 if it's rare.
        # To be efficient, we'll calculate document frequency of the pool.
        pass # Placeholder for logic below

    # Refined pass: because we need Pool Doc Frequency, we need a two-pass approach
    # but we are limited by 180s. Let's simplify the "doc frequency" to 
    # "appears in < 3 docs" by scanning the whole pool once.
    pool_word_counts = Counter()
    for doc in docs:
        w_list = re.findall(r'[a-z]+', doc.lower())
        unique_words = set(w_list)
        for w in unique_words:
            pool_word_counts[w] += 1

    doc_features = []
    for i, doc in enumerate(docs):
        norm_text = " ".join(doc.split())
        f_hash = hash(norm_text)
        p_hash = hash(norm_text[:300])
        if f_hash in seen_full or p_hash in seen_partial:
            continue
        seen_full.add(f_hash)
        seen_partial.add(p_hash)

        chars = list(doc)
        num_chars = len(doc)
        is_wiki_start = doc.startswith('= ')
        wiki_markers = [' , ', ' . ', ' @-@ ', ' @,@ ', ' ( ']
        marker_count = sum(doc.count(m) for m in wiki_markers)
        is_wiki = is_wiki_start and (marker_count / num_chars >= 0.01 if num_chars > 0 else False)

        # Rare word rate
        words = re.findall(r'[a-z]+', doc.lower())
        alpha_words = [w for w in words if w.isalpha()]
        rare_count = 0
        for w in alpha_words:
            if pool_word_counts[w] <= 2 and w not in target_vocab:
                rare_count += 1
        rare_rate = rare_count / len(alpha_words) if alpha_words else 0

        # Stopword bigram rate
        bigrams = []
        for j in range(len(words) - 1):
            bigrams.append((words[j], words[j+1]))
        bigram_rate = sum(1 for b in bigrams if b in target_bigrams) / len(bigrams) if bigrams else 0

        # Zlib
        z_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc.encode('utf-8')) if len(doc) > 0 else 1.0
        
        # Mojibake
        mojibake = sum(1 for c in doc if c in ('Ã', 'â', '€', 'Â'))
        
        # Metadata
        doc_features.append({
            'id': i,
            'is_wiki': is_wiki,
            'rare_rate': rare_rate,
            'bigram_rate': bigram_rate,
            'z_ratio': z_ratio,
            'mojibake': mojibake > 0,
            'tokens': len(enc.encode(doc))
        })
        if is_wiki:
            wiki_rare_word_rates.append(rare_rate)
            wiki_bigram_rates.append(bigram_rate)

    # Stats
    wiki_rare_median = np.median(wiki_rare_word_rates) if wiki_rare_word_rates else 0
    wiki_rare_mad = np.mad_std(wiki_rare_word_rates) if len(wiki_rare_word_rates) > 0 else 0
    wiki_bigram_median = np.median(wiki_bigram_rates) if wiki_bigram_rates else 0
    wiki_bigram_mad = np.mad_std(wiki_bigram_rates) if len(wiki_bigram_rates) > 0 else 0

    # Tiering
    tier_a = []
    tier_b = []
    tier_c = []

    for df in doc_features:
        if df['is_wiki']:
            if (df['rare_rate'] < (wiki_rare_median + 2 * wiki_rare_mad) and 
                df['bigram_rate'] > (wiki_bigram_median - 2 * wiki_bigram_mad) and 
                df['z_ratio'] > 0.25):
                tier_a.append(df)
            else:
                tier_b.append(df)
        else:
            if (df['z_ratio'] > 0.3 and not df['mojibake'] and 
                not any(x in doc for x in ['Click here', 'best price', ' | '])):
                tier_c.append(df)

    # Sorting
    tier_b.sort(key=lambda x: x['rare_rate'])
    tier_c.sort(key=lambda x: x['bigram_rate'], reverse=True)

    # Selection
    selected_indices = []
    total_tokens = 0
    target_count = int(1.05 * T)
    
    for tier in [tier_a, tier_b, tier_c]:
        for df in tier:
            selected_indices.append(df['id'])
            total_tokens += df['tokens']
            if total_tokens >= target_count:
                break
        if total_tokens >= target_count:
            break

    # Shuffle final result
    rng.shuffle(selected_indices)
    return selected_indices
