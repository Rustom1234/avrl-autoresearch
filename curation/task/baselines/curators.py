"""All no-LLM baselines and the label oracles. Mode comes from env CURATE_MODE (see run_pilot.py).
Oracles read the hidden labels from env CURATION_ORACLE_LABELS: analysis only, never available to a real curator."""
import hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import T
import numpy as np
import tiktoken

MODE = os.environ["CURATE_MODE"]
RNG = random.Random(int(os.environ.get("CURATE_SEED", 0)))
ENC = tiktoken.get_encoding("gpt2")


def ntok(docs):
    return np.array([len(t) + 1 for t in ENC.encode_ordinary_batch(docs, num_threads=16)])


def shuffled(ids):
    ids = list(ids); RNG.shuffle(ids); return ids


def epochs(ids, n=4):
    """n independent shuffles of ids, concatenated (the runner cuts at T tokens)."""
    out = []
    for _ in range(n):
        out += shuffled(ids)
    return out


def junk(d):
    if len(d) < 200:
        return True
    lines = [l for l in d.split("\n") if l.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:           # repeated boilerplate lines
        return True
    if sum(1 for l in lines if len(l) >= 60 and " " not in l.strip()) / max(1, len(lines)) > 0.3:  # base64-like
        return True
    if sum(c.isdigit() for c in d) / len(d) > 0.3:
        return True
    return False


def select(docs, target_sample):
    n = len(docs)
    if MODE == "dedup":
        seen, keep = set(), []
        for i, d in enumerate(docs):
            h = hashlib.md5(d.strip().encode()).hexdigest()
            if h not in seen:
                seen.add(h); keep.append(i)
        return shuffled(keep)
    if MODE == "random50":
        return epochs(RNG.sample(range(n), n // 2))
    if MODE == "heuristic":
        return shuffled(i for i, d in enumerate(docs) if not junk(d))
    if MODE.startswith("tfidf"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        k = float(MODE[5:])
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, stop_words="english", dtype=np.float32)
        X = vec.fit_transform(docs)
        c = np.asarray(vec.transform(target_sample).mean(axis=0)).ravel()
        sims = X @ (c / np.linalg.norm(c))
        order = np.argsort(-sims)
        cum = np.cumsum(ntok(docs)[order])
        keep = order[: int(np.searchsorted(cum, k * T)) + 1]
        return epochs(keep.tolist())
    if MODE.startswith("oracle"):
        kind = [l["kind"] for l in json.load(open(os.environ["CURATION_ORACLE_LABELS"]))]
        wiki = [i for i, k in enumerate(kind) if k == "clean_wiki"]
        web = [i for i, k in enumerate(kind) if k == "clean_web"]
        if MODE == "oracle_wiki":
            return epochs(wiki)
        if MODE == "oracle_wiki_web":
            return epochs(wiki + web)
        if MODE == "oracle_web_wiki":
            return shuffled(web) + epochs(wiki)
    raise ValueError(MODE)
