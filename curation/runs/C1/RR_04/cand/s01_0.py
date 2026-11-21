"""Curator: drop obvious junk and duplicates, then prioritise documents whose
surface structure (punctuation mix, character classes, word length, document
length) looks like the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import zlib
import math

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

CAP = 3000  # characters of each document inspected for features

SPAM_PHRASES = [
    "click here to order", "free shipping", "click here", "buy now",
    "order now", "add to cart", "limited time offer", "subscribe now",
    "unsubscribe", "all rights reserved", "terms and conditions",
    "privacy policy", "sign up today", "money back guarantee",
    "lorem ipsum", "viagra", "casino", "best price", "discount code",
    "act now", "call now", "100% free", "special offer",
]

PUNCT = [b".", b",", b";", b":", b"'", b'"', b"(", b")", b"-", b"!", b"?",
         b"=", b"|", b"{", b"}", b"<", b">", b"/", b"[", b"]", b"*", b"#",
         b"_", b"@", b"%", b"$", b"&", b"\\", b"\n", b" ", b"\t"]

_LOWER = bytes(range(ord("a"), ord("z") + 1))
_UPPER = bytes(range(ord("A"), ord("Z") + 1))
_DIGIT = b"0123456789"
_HIGH = bytes(range(128, 256))


def _features(doc):
    """Return (feature list, alpha fraction, compression ratio)."""
    s = doc[:CAP]
    b = s.encode("utf-8", "ignore")
    n = len(b)
    if n == 0:
        return None
    inv = 1.0 / n
    lower = (n - len(b.translate(None, _LOWER))) * inv
    upper = (n - len(b.translate(None, _UPPER))) * inv
    digit = (n - len(b.translate(None, _DIGIT))) * inv
    high = (n - len(b.translate(None, _HIGH))) * inv
    feats = [lower, upper, digit, high]
    for p in PUNCT:
        feats.append(b.count(p) * inv)
    words = s.split()
    nw = len(words)
    feats.append(min(len(s) / (nw + 1.0), 30.0))            # mean word length
    feats.append(len(set(words)) / (nw + 1.0))               # lexical variety
    lines = s.count("\n") + 1
    feats.append(math.log(len(s) / lines + 1.0))             # mean line length
    comp = len(zlib.compress(b, 1)) / float(n)
    feats.append(comp)                                       # compressibility
    feats.append(math.log(len(doc) + 1.0))                   # document length
    return feats, lower + upper + high, comp


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)
    if n_docs == 0:
        return []

    # ---- target statistics -------------------------------------------------
    tgt = [t for t in target_sample if isinstance(t, str) and t.strip()]
    tfeats, talpha, tcomp, tlens = [], [], [], []
    for t in tgt:
        r = _features(t)
        if r is None:
            continue
        tfeats.append(r[0])
        talpha.append(r[1])
        tcomp.append(r[2])
        tlens.append(len(t.strip()))
    have_target = len(tfeats) >= 5

    if have_target:
        tf = np.asarray(tfeats, dtype=np.float64)
        mu = tf.mean(axis=0)
        sd = tf.std(axis=0)
        sd = np.maximum(sd, 0.05 * np.abs(mu) + 1e-3)
        weights = np.ones(tf.shape[1])
        weights[-1] = 0.5  # document length: only a mild preference
        min_len = max(30, int(0.3 * min(tlens)))
        alpha_floor = min(0.3, 0.5 * float(np.mean(talpha)))
        comp_floor = min(0.12, 0.5 * float(np.min(tcomp)))
        target_lower = "\n".join(tgt).lower()
        spam = [p for p in SPAM_PHRASES if p not in target_lower]
    else:
        min_len, alpha_floor, comp_floor = 30, 0.3, 0.12
        spam = list(SPAM_PHRASES)

    # ---- scan the pool -----------------------------------------------------
    seen = set()
    good_idx, good_feats = [], []
    junk_idx, junk_feats = [], []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        st = d.strip()
        if not st:
            continue
        key = hash((len(st) // 64, " ".join(st[:CAP].lower().split())))
        if key in seen:
            continue  # duplicate (exact or near-exact)
        seen.add(key)
        r = _features(st)
        if r is None:
            continue
        feats, alpha, comp = r
        bad = False
        if len(st) < min_len:
            bad = True
        elif alpha < alpha_floor:
            bad = True
        elif comp < comp_floor and len(st) > 400:
            bad = True  # extremely repetitive
        elif st.count("\ufffd") > 0.005 * len(st) + 2:
            bad = True  # encoding damage
        else:
            low = st[:20000].lower()
            for p in spam:
                if p in low:
                    bad = True
                    break
        if bad:
            junk_idx.append(i)
            junk_feats.append(feats)
        else:
            good_idx.append(i)
            good_feats.append(feats)

    def score(feat_list):
        if not feat_list:
            return np.zeros(0)
        if not have_target:
            return np.zeros(len(feat_list))
        x = np.asarray(feat_list, dtype=np.float64)
        z = (x - mu) / sd
        z2 = np.minimum(z * z, 25.0)
        return -(z2 * weights).sum(axis=1) / weights.sum()

    good_scores = score(good_feats)
    order = np.argsort(-good_scores, kind="stable")
    ranked = [good_idx[j] for j in order]

    # ---- estimate tokens ---------------------------------------------------
    chars_per_tok = 4.0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        samp = rng.sample(ranked, min(200, len(ranked))) if ranked else []
        texts = [docs[i][:2000] for i in samp]
        nchar = sum(len(t) for t in texts)
        ntok = sum(len(x) for x in enc.encode_ordinary_batch(texts))
        if ntok > 0 and nchar > 0:
            chars_per_tok = max(1.5, min(6.0, nchar / float(ntok)))
    except Exception:
        pass

    # ---- take the best-matching documents up to the budget ------------------
    budget = 1.05 * T
    acc = 0.0
    cut = len(ranked)
    for k, i in enumerate(ranked):
        acc += len(docs[i]) / chars_per_tok + 1.0
        if acc >= budget:
            cut = k + 1
            break
    selected = ranked[:cut]
    rest = ranked[cut:]
    rng.shuffle(selected)  # mix so training is not ordered by score

    result = selected + rest

    # Safety net: if filtering left very little, fall back on rejected docs
    # (best-matching first) rather than cycling a tiny set many times.
    total_good = sum(len(docs[i]) for i in ranked) / chars_per_tok
    if total_good < 0.3 * T and junk_idx:
        js = score(junk_feats)
        jorder = np.argsort(-js, kind="stable")
        result += [junk_idx[j] for j in jorder]

    if not result:
        result = list(range(n_docs))
        rng.shuffle(result)
    return result
