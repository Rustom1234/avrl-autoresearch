import os, random, re


def _words(text, n=300):
    return re.findall(r"\S+", text.lower())[:n]


def _bigrams(words):
    return list(zip(words, words[1:]))


def select(docs, target_sample):
    from common import T

    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Build the set of word bigrams from the target sample
    target_bigrams = set()
    for t in target_sample:
        target_bigrams.update(_bigrams(_words(t, 10**9)))

    # Score each document
    scores = []
    for i, d in enumerate(docs):
        bg = _bigrams(_words(d))
        if not bg:
            s = 0.0
        else:
            hits = sum(1 for b in bg if b in target_bigrams)
            s = hits / len(bg)
        scores.append((s, i))

    # Highest scores first; ties broken by seeded random order
    tiebreak = list(range(len(docs)))
    rng.shuffle(tiebreak)
    order = sorted(scores, key=lambda x: (-x[0], tiebreak[x[1]]))

    kept = []
    total = 0
    for s, i in order:
        if total >= T:
            break
        kept.append(i)
        total += len(docs[i]) / 4

    rng.shuffle(kept)
    return kept
