import os, random


def _keep(doc):
    lines = [l.strip() for l in doc.split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return False
    # Mostly repeated lines
    if len(set(lines)) / len(lines) < 0.5:
        return False
    # Too many long strings without spaces (e.g. base64, URLs, hashes)
    long_nospace = sum(1 for l in lines if len(l) >= 40 and not any(c.isspace() for c in l))
    if long_nospace / len(lines) > 0.30:
        return False
    return True


def select(docs, target_sample):
    idx = [i for i, d in enumerate(docs) if _keep(d)]
    if not idx:
        idx = list(range(len(docs)))
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
    return idx
