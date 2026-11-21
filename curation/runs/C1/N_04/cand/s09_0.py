"""Written by the no-model menu search (curation/loop/menu.py)."""
import hashlib, os, random
import numpy as np

CFG = {'dedup': True, 'junk': False, 'rank': 'word', 'keep': 1.5}
T = 9830400
RNG = random.Random(int(os.environ.get("CURATE_SEED", 0)))


def junk(d):
    if len(d) < 200:
        return True
    lines = [l for l in d.split("\n") if l.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:
        return True
    if sum(1 for l in lines if len(l) >= 60 and " " not in l.strip()) / max(1, len(lines)) > 0.3:
        return True
    return sum(c.isdigit() for c in d) / len(d) > 0.3


def select(docs, target_sample):
    ids = list(range(len(docs)))
    if CFG["dedup"]:
        seen, keep = set(), []
        for i in ids:
            h = hashlib.md5(docs[i].strip().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                keep.append(i)
        ids = keep
    if CFG["junk"]:
        ids = [i for i in ids if not junk(docs[i])]
    if CFG["rank"] != "none":
        import tiktoken
        from sklearn.feature_extraction.text import TfidfVectorizer
        if CFG["rank"] == "word":
            vec, cut = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, stop_words="english",
                                       dtype=np.float32), None
        else:
            vec, cut = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2,
                                       max_features=50000, dtype=np.float32), 1000
        X = vec.fit_transform([docs[i][:cut] for i in ids])
        c = np.asarray(vec.transform([t[:cut] for t in target_sample]).mean(axis=0)).ravel()
        order = np.argsort(-(X @ (c / np.linalg.norm(c))))
        enc = tiktoken.get_encoding("gpt2")
        ntok = np.array([len(t) + 1 for t in enc.encode_ordinary_batch([docs[ids[j]] for j in order], num_threads=8)])
        ids = [ids[j] for j in order[: int(np.searchsorted(np.cumsum(ntok), CFG["keep"] * T)) + 1]]
    out = []
    for _ in range(4):
        e = list(ids)
        RNG.shuffle(e)
        out += e
    return out
