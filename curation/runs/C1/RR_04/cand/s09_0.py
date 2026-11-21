import os
import random
import re
import zlib

from common import T

SPAM_PHRASES = (
    'click here to order', 'buy now', 'free shipping',
    'discount code', 'coupon', 'for sale |',
    'limited offer', 'best price',
)
PRE_THE = frozenset({'of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from',
                     'with', 'as', 'that', 'was', 'is'})
BAD_PAIRS = frozenset({('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')})
WORD4_RE = re.compile(r'[a-z]{4,}')
ALPHA_RE = re.compile(r'[a-zA-Z]+')


def _passes_filters(doc, dl, tks, low, V):
    # Spam / template check
    if len(doc) < 200:
        return False
    if len(zlib.compress(doc.encode('utf-8'), 1)) / len(doc) < 0.30:
        return False
    for p in SPAM_PHRASES:
        if p in dl:
            return False

    # Letter-scramble check
    scram = WORD4_RE.findall(dl)
    if len(scram) >= 20:
        m = sum(1 for w in scram if w in V)
        if m / len(scram) < 0.5:
            return False

    n = len(tks)
    if n == 0:
        return True

    # Word-shuffle check (a): what precedes 'the'
    tc = 0
    ptc = 0
    prev = None
    for w in low:
        if w == 'the':
            tc += 1
            if prev in PRE_THE:
                ptc += 1
        prev = w
    if tc >= 10 and (ptc / tc) < 0.25:
        return False

    # (b): sentence end followed by a lowercase word
    dc = 0
    dfl = 0
    for j in range(n - 1):
        if tks[j].endswith('.'):
            dc += 1
            if tks[j + 1][0].islower():
                dfl += 1
    if dc >= 8 and (dfl / dc) > 0.4:
        return False

    # (c): adjacent function-word pairs
    pc = 0
    for j in range(n - 1):
        a = low[j]
        if a == 'the' or a == 'of' or a == 'a':
            if (a, low[j + 1]) in BAD_PAIRS:
                pc += 1
            if (a == 'the' and j + 3 < n and low[j + 1] == 'of'
                    and low[j + 2] == 'the' and low[j + 3] == 'the'):
                pc += 1
    if (pc / n) > 0.005:
        return False

    return True


def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts)]

        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]

        return count


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)  # seeded; the selection itself is deterministic

    # Vocabularies from the target sample
    V = set()        # alphabetic words (for the scramble check)
    V_tok = set()    # whitespace tokens (for the vocabulary similarity score)
    for doc in target_sample:
        V.update(ALPHA_RE.findall(doc.lower()))
        V_tok.update(doc.lower().split())

    survivors = []  # (density, vocab_sim, index)
    seen_hashes = set()

    for i, doc in enumerate(docs):
        if len(doc) < 200:
            continue
        dl = doc.lower()
        tks = doc.split()
        low = [t.lower() for t in tks]
        if not _passes_filters(doc, dl, tks, low, V):
            continue

        # Dedupe on whitespace-normalised text
        nh = hash(" ".join(tks))
        if nh in seen_hashes:
            continue
        seen_hashes.add(nh)

        # Wiki-density
        markers = (doc.count('= ') + doc.count(' , ') + doc.count(' . ')
                   + doc.count(' @-@ '))
        density = markers / len(doc)

        # Vocabulary similarity score over the first 500 tokens
        head = low[:500]
        if head:
            vocab_sim = sum(1 for w in head if w in V_tok) / len(head)
        else:
            vocab_sim = 0.0

        survivors.append((density, vocab_sim, i))

    if not survivors:
        return list(range(len(docs)))

    # Keep documents whose density is above the median of all survivors
    dens_sorted = sorted(s[0] for s in survivors)
    n = len(dens_sorted)
    if n % 2:
        median = dens_sorted[n // 2]
    else:
        median = 0.5 * (dens_sorted[n // 2 - 1] + dens_sorted[n // 2])
    kept = [s for s in survivors if s[0] > median]
    if not kept:
        kept = survivors

    # Sort by density (desc), then vocabulary similarity (desc), then index
    kept.sort(key=lambda x: (-x[0], -x[1], x[2]))

    # Select until the cumulative (real) token count reaches T
    count_tokens = _token_counter()
    selected = []
    total = 0
    batch = 512
    for start in range(0, len(kept), batch):
        chunk = kept[start:start + batch]
        counts = count_tokens([docs[idx] for _, _, idx in chunk])
        for (_, _, idx), c in zip(chunk, counts):
            selected.append(idx)
            total += c
            if total >= T:
                break
        if total >= T:
            break

    # If fewer than T tokens were selected, the trainer cycles the list.
    return selected
