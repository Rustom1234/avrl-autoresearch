import os, random, re


def _bigrams(text, limit=300):
    words = re.findall(r"\S+", text.lower())[:limit]
    return zip(words, words[1:])


def select(docs, target_sample):
    from common import T

    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target = set()
    for t in target_sample:
        target.update(_bigrams(t, 10**9))

    scores = []
    for i, d in enumerate(docs):
        bg = list(_bigrams(d))
        if not bg:
            s = 0.0
        else:
            s = sum(1 for b in bg if b in target) / len(bg)
        scores.append((s, i))

    scores.sort(key=lambda x: (-x[0], x[1]))

    kept = []
    total = 0
    for s, i in scores:
        if total >= T:
            break
        n = len(docs[i]) / 4
        if n <= 0:
            continue
        kept.append(i)
        total += n

    rng.shuffle(kept)
    return kept
