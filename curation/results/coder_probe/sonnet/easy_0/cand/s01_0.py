import os, random

LONG_NOSPACE_LEN = 50


def _keep(doc):
    lines = [l.strip() for l in doc.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return False
    n = len(lines)
    if len(set(lines)) / n < 0.5:
        return False
    long_ns = sum(1 for l in lines if len(l) >= LONG_NOSPACE_LEN and not any(c.isspace() for c in l))
    if long_ns / n > 0.3:
        return False
    return True


def select(docs, target_sample):
    idx = [i for i, d in enumerate(docs) if _keep(d)]
    if not idx:
        idx = list(range(len(docs)))
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
    return idx
