import os, random


def _keep(doc):
    lines = [l.strip() for l in doc.split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return False
    # Unique-line ratio below 0.5 -> mostly repeats
    if len(set(lines)) / len(lines) < 0.5:
        return False
    # More than 30% of lines are long strings without spaces
    long_nospace = sum(1 for l in lines if len(l) > 40 and " " not in l)
    if long_nospace / len(lines) > 0.3:
        return False
    return True


def select(docs, target_sample):
    idx = [i for i, d in enumerate(docs) if _keep(d)]
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
    return idx
