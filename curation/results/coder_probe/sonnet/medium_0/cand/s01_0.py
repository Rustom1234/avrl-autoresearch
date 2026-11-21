import os, random, re


def _words(text):
    return re.findall(r"\w+", text.lower())[:300]


def _bigrams(words):
    return list(zip(words, words[1:]))


def select(docs, target_sample):
    from common import T

    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Build the set of word bigrams from the target sample
    target_bigrams = set()
    for t in target_sample:
        w = re.findall(r"\w+", t.lower())
        target_bigrams.update(zip(w, w[1:]))

    # Score each document
    scores = []
    for i, d in enumerate(docs):
        bg = _bigrams(_words(d))
        if bg:
            s = sum(1 for b in bg if b in target_bigrams) / len(bg)
        else:
            s = 0.0
        scores.append((s, i))

    # Highest scores first; ties broken by random order for determinism per seed
    tiebreak = list(range(len(docs)))
    rng.shuffle(tiebreak)
    order = sorted(range(len(docs)), key=lambda i: (-scores[i][0], tiebreak[i]))

    kept = []
    total = 0
    for i in order:
        est = len(docs[i]) / 4
        if est <= 0:
            continue
        kept.append(i)
        total += est
        if total >= T:
            break

    rng.shuffle(kept)
    return kept
