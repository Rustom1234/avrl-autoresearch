import os
import random
import zlib
import re
from collections import Counter
from common import T


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Vocabulary of the target sample (alphabetic words) and its common subset
    V = set()
    doc_freq = Counter()
    for doc in target_sample:
        words = re.findall(r'[a-zA-Z]+', doc.lower())
        V.update(words)
        doc_freq.update(set(words))
    # "Common" words: the 3000 words that appear in the most target documents
    common = set(w for w, _ in doc_freq.most_common(3000))
    if not common:
        common = V

    spam_phrases = [
        'click here to order', 'buy now', 'free shipping',
        'discount code', 'coupon', 'for sale |',
        'limited offer', 'best price'
    ]
    pre_words = {'of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from',
                 'with', 'as', 'that', 'was', 'is'}
    bad_pairs = {('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')}
    word_re = re.compile(r'[a-zA-Z]+')

    kept_indices = []
    scores = {}
    seen_full_hashes = set()
    seen_prefix_hashes = set()

    for i, doc in enumerate(docs):
        # Spam and template check
        if len(doc) < 200:
            continue

        compressed_ratio = len(zlib.compress(doc.encode('utf-8'))) / len(doc)
        if compressed_ratio < 0.30:
            continue

        doc_lower = doc.lower()
        if any(phrase in doc_lower for phrase in spam_phrases):
            continue

        # Letter-scramble check
        alpha_words = word_re.findall(doc_lower)
        scramble_tokens = [w for w in alpha_words if len(w) >= 4]
        if len(scramble_tokens) >= 20:
            match_count = sum(1 for w in scramble_tokens if w in V)
            if match_count / len(scramble_tokens) < 0.5:
                continue

        # Word-shuffle check
        tokens = doc.split()
        if len(tokens) > 0:
            ltokens = [t.lower() for t in tokens]
            n = len(tokens)

            # (a) token preceding 'the'
            pre_the_count = 0
            the_count = 0
            for j in range(n):
                if ltokens[j] == 'the':
                    the_count += 1
                    if j > 0 and ltokens[j - 1] in pre_words:
                        pre_the_count += 1
            if the_count >= 10 and (pre_the_count / the_count) < 0.25:
                continue

            # (b) sentence-final '.' followed by a lowercase word
            dot_count = 0
            dot_followed_lower = 0
            for j in range(n - 1):
                if tokens[j].endswith('.'):
                    dot_count += 1
                    next_t = tokens[j + 1]
                    if next_t and next_t[0].islower():
                        dot_followed_lower += 1
            if dot_count >= 8 and (dot_followed_lower / dot_count) > 0.4:
                continue

            # (c) suspicious adjacent function-word patterns
            pair_count = 0
            for j in range(n - 3):
                if (ltokens[j] == 'the' and ltokens[j + 1] == 'of'
                        and ltokens[j + 2] == 'the' and ltokens[j + 3] == 'the'):
                    pair_count += 1
            for j in range(n - 1):
                if (ltokens[j], ltokens[j + 1]) in bad_pairs:
                    pair_count += 1
            if (pair_count / n) > 0.005:
                continue

        # Dedupe
        norm_text = " ".join(tokens)
        h1 = hash(norm_text)
        h2 = hash(norm_text[:300])
        if h1 in seen_full_hashes or h2 in seen_prefix_hashes:
            continue
        seen_full_hashes.add(h1)
        seen_prefix_hashes.add(h2)

        # High-quality score: common target words / total tokens
        common_count = sum(1 for w in alpha_words if w in common)
        denom = max(len(tokens), len(alpha_words), 1)
        scores[i] = common_count / denom
        kept_indices.append(i)

    # Sort survivors by score, highest first (seeded shuffle breaks ties)
    rng.shuffle(kept_indices)
    kept_indices.sort(key=lambda idx: scores[idx], reverse=True)

    # Token counter: tiktoken if available, otherwise a character estimate
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    budget = int(T * 1.02)  # small margin over T
    selected = []
    total = 0
    chunk = 512
    pos = 0
    while pos < len(kept_indices) and total < budget:
        batch = kept_indices[pos:pos + chunk]
        pos += chunk
        lengths = None
        if enc is not None:
            try:
                encoded = enc.encode_ordinary_batch([docs[i] for i in batch])
                lengths = [len(e) for e in encoded]
            except Exception:
                lengths = None
        if lengths is None:
            lengths = [max(1, len(docs[i]) // 4) for i in batch]
        for idx, ln in zip(batch, lengths):
            selected.append(idx)
            total += ln + 1
            if total >= budget:
                break

    # Shuffle the selected set so training does not see a score-ordered stream
    rng.shuffle(selected)
    return selected
