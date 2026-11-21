"""Rebuild the noisy pool and the target-domain splits from a seed.

    python build_pool.py --preset v1 --seed 0      -> data/curation/pool_v1/
Inputs (all local after the first download, see README): wikitext-103-raw parquet,
fineweb sample, github-code shard, four StackExchange jsonl.gz dumps.
"""
import argparse, gzip, html, json, random, re
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from common import ROOT, DATA, T, MAX_CHARS

# token shares of the pool (sum to 1). wiki share fixes pool size: P = WIKI_CLEAN_T / share["wiki"].
PRESETS = {
    "v1": dict(wiki=0.25, web=0.20, code=0.15, qa=0.10, dup_wiki=0.075, dup_web=0.075,
               ch_shuf=0.03, boiler=0.03, b64=0.02, wordshuf=0.04, spam=0.03),
    # harder: same 0.4T of clean wiki, but half the pool is junk and 2/3 of the junk is HARD (wordshuf + spam)
    "v2": dict(wiki=0.20, web=0.12, code=0.08, qa=0.05, dup_wiki=0.06, dup_web=0.04,
               ch_shuf=0.03, boiler=0.03, b64=0.02, wordshuf=0.19, spam=0.18),
}
WIKI_CLEAN_T = 0.4 * T
N_TARGET, N_DEV, N_SEALED = 200, 150, 150
BOILER = ["Copyright (c) 2013 All rights reserved. Terms of use | Privacy policy | Contact us | Sitemap",
          "Click here to subscribe to our newsletter and get the latest updates in your inbox",
          "Home | About | Products | Services | Login | Register | Cart (0 items)",
          "Posted by admin in Uncategorized. Leave a comment. Read more >>"]
SPAM_T = ["Best {a} {b} online store | cheap {c} for sale | buy {a} {d} now with free shipping on all {e} orders. ",
          "Looking for {a} {b}? Click here to order {c} {d} today and save 70%! Limited offer on {e}. ",
          "{A} {B} {C} discount code | {D} {E} coupon | top rated {a} {c} deals, buy now. ",
          "Get {a} {b} {c} cheap! Order {d} online, fast delivery, best price guaranteed for {e}. "]


def trunc(s, n=MAX_CHARS):
    if len(s) <= n:
        return s
    cut = s.rfind("\n", n // 2, n)
    if cut < 0:
        cut = s.rfind(" ", n // 2, n)
    return s[: cut if cut > 0 else n]


def read_wiki():
    arts, cur = [], []
    for sh in (0, 1):
        col = pq.read_table(DATA / f"raw/wikitext-103-raw-v1/train-0000{sh}-of-00002.parquet")["text"].to_pylist()
        for ln in col:
            ln = ln.strip()
            if not ln:
                continue
            if re.fullmatch(r"= [^=].* =", ln):  # level-1 header starts an article
                if cur:
                    arts.append("\n".join(cur))
                cur = []
            cur.append(ln)
    if cur:
        arts.append("\n".join(cur))
    return [trunc(a) for a in arts if len(a) >= 1500]


def read_web(rng, n_groups=14):
    pf = pq.ParquetFile(ROOT / "data/fineweb/sample/10BT/000_00000.parquet")
    out = []
    for g in rng.sample(range(pf.metadata.num_row_groups), n_groups):
        out += [trunc(t) for t in pf.read_row_group(g, columns=["text"])["text"].to_pylist() if len(t) >= 300]
    rng.shuffle(out)
    return out


def read_code(rng):
    pf = pq.ParquetFile(ROOT / "data/github-code/data/train-00000-of-00880.parquet")
    out = []
    for g in rng.sample(range(pf.metadata.num_row_groups), 3):
        out += [trunc(t) for t in pf.read_row_group(g, columns=["code"])["code"].to_pylist() if len(t) >= 300]
    rng.shuffle(out)
    return out


def read_qa(rng):
    out = []
    for site in ["stats", "unix", "physics", "cooking"]:
        with gzip.open(DATA / f"raw/{site}.stackexchange.com.jsonl.gz", "rt") as f:
            for i, ln in enumerate(f):
                if rng.random() < 0.05:
                    q, a = json.loads(ln)[:2]
                    t = html.unescape(q + "\n\n" + a)
                    if len(t) >= 300:
                        out.append(trunc(t))
    rng.shuffle(out)
    return out


def near_dup(s, rng):
    w = s.split(" ")
    a = rng.randrange(0, max(1, int(len(w) * 0.8)))
    b = a + int(len(w) * rng.uniform(0.1, 0.2))
    w = w[:a] + w[b:]
    for i in rng.sample(range(len(w)), max(1, len(w) // 40)):  # ~2.5% typos
        if len(w[i]) > 3:
            j = rng.randrange(len(w[i]))
            w[i] = w[i][:j] + w[i][j + 1:]
    return " ".join(w)


def word_shuffle(s, rng):
    lines = []
    for ln in s.split("\n"):
        if ln.startswith("="):
            lines.append(ln)
            continue
        sents = []
        for sent in ln.split(" . "):
            w = sent.split(" ")
            rng.shuffle(w)
            sents.append(" ".join(w))
        lines.append(" . ".join(sents))
    return "\n".join(lines)


def char_shuffle(s, rng):
    def f(m):
        c = list(m.group(0)); rng.shuffle(c); return "".join(c)
    return re.sub(r"\w+", f, s)


def spam_doc(vocab, rng):
    target = rng.randint(1500, 5000)
    out, n = [], 0
    while n < target:
        d = {k: rng.choice(vocab) for k in "abcde"}
        d.update({k.upper(): d[k].capitalize() for k in "abcde"})
        s = rng.choice(SPAM_T).format(**d)
        out.append(s); n += len(s)
    return "".join(out)


def boiler_doc(web, rng):
    lines = rng.sample(BOILER, rng.randint(1, 2))
    target = rng.randint(1500, 5000)
    out, n = [], 0
    while n < target:
        ln = rng.choice(lines) if rng.random() < 0.9 else rng.choice(web)[:120].replace("\n", " ")
        out.append(ln); n += len(ln) + 1
    return "\n".join(out)


def b64_doc(rng):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return "\n".join("".join(rng.choices(chars, k=76)) for _ in range(rng.randint(30, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sh = PRESETS[args.preset]
    assert abs(sum(sh.values()) - 1) < 1e-9
    rng = random.Random(args.seed)
    enc = tiktoken.get_encoding("gpt2")
    ntok = lambda texts: [len(t) for t in enc.encode_ordinary_batch(texts, num_threads=16)]
    P = WIKI_CLEAN_T / sh["wiki"]

    wiki = read_wiki(); rng.shuffle(wiki)
    target_sample, dev, sealed = wiki[:N_TARGET], wiki[N_TARGET:N_TARGET + N_DEV], wiki[N_TARGET + N_DEV:N_TARGET + N_DEV + N_SEALED]
    rest = wiki[N_TARGET + N_DEV + N_SEALED:]   # disjoint from all three splits
    docs, labels = [], []

    def take(pool, kind, budget, fn=lambda s: s):
        tot, ids = 0, []
        while tot < budget:
            d = fn(pool.pop(0))
            ids.append(len(docs)); docs.append(d); labels.append(dict(kind=kind, src=None)); tot += ntok([d])[0]
        return ids

    wiki_ids = take(rest, "clean_wiki", sh["wiki"] * P)
    web_src, code_src, qa_src = read_web(rng), read_code(rng), read_qa(rng)
    web_ids = take(web_src, "clean_web", sh["web"] * P)
    take(code_src, "code", sh["code"] * P)
    take(qa_src, "qa", sh["qa"] * P)
    # duplicates of pool wiki / web docs, half exact, half near
    for name, ids in (("wiki", wiki_ids), ("web", web_ids)):
        budget, tot, i = sh["dup_" + name] * P, 0, 0
        order = rng.sample(ids, len(ids))
        while tot < budget:
            s_id = order[i % len(order)]; i += 1
            exact = i % 2 == 1
            d = docs[s_id] if exact else near_dup(docs[s_id], rng)
            docs.append(d); labels.append(dict(kind=f"dup_{'exact' if exact else 'near'}_{name}", src=s_id)); tot += ntok([d])[0]
    # damaged
    take(web_src, "dmg_charshuf", sh["ch_shuf"] * P, lambda s: char_shuffle(s, rng))
    for kind, key, mk in (("dmg_boiler", "boiler", lambda: boiler_doc(web_src[:50], rng)), ("dmg_b64", "b64", lambda: b64_doc(rng))):
        tot = 0
        while tot < sh[key] * P:
            d = mk(); docs.append(d); labels.append(dict(kind=kind, src=None)); tot += ntok([d])[0]
    take(rest, "dmg_wordshuf", sh["wordshuf"] * P, lambda s: word_shuffle(s, rng))
    words = re.findall(r"[a-z]{4,}", " ".join(rest[:300]).lower())
    tot = 0
    while tot < sh["spam"] * P:
        d = spam_doc(words, rng); docs.append(d); labels.append(dict(kind="dmg_spam", src=None)); tot += ntok([d])[0]

    # shuffle pool order, remap dup sources
    perm = list(range(len(docs))); rng.shuffle(perm)
    inv = {old: new for new, old in enumerate(perm)}
    docs2 = [docs[o] for o in perm]
    labels2 = [dict(kind=labels[o]["kind"], src=None if labels[o]["src"] is None else inv[labels[o]["src"]]) for o in perm]

    out = DATA / f"pool_{args.preset}"; out.mkdir(parents=True, exist_ok=True)
    json.dump(docs2, open(out / "docs.json", "w"))
    json.dump(labels2, open(out / "labels_HIDDEN.json", "w"))
    json.dump(target_sample, open(out / "target_sample.json", "w"))
    toks = enc.encode_ordinary_batch(docs2, num_threads=16)
    lens = np.array([len(t) + 1 for t in toks], dtype=np.int64)       # +1 for EOT appended in training
    flat = np.concatenate([np.array(t + [enc.eot_token], dtype=np.uint16) for t in toks])
    np.save(out / "pool_tokens.npy", flat); np.save(out / "pool_offsets.npy", np.concatenate([[0], np.cumsum(lens)]))
    for name, ds in (("dev", dev), ("sealed", sealed)):
        tt = enc.encode_ordinary_batch(ds, num_threads=16)
        np.save(out / f"{name}_tokens.npy", np.concatenate([np.array(t + [enc.eot_token], dtype=np.uint16) for t in tt]))
    # summary
    kinds = {}
    for l, n in zip(labels2, lens):
        kinds[l["kind"]] = kinds.get(l["kind"], 0) + int(n)
    summ = dict(preset=args.preset, seed=args.seed, n_docs=len(docs2), pool_tokens=int(lens.sum()), pool_over_T=float(lens.sum() / T),
                tokens_by_kind_over_T={k: round(v / T, 4) for k, v in sorted(kinds.items())},
                dev_tokens=int(sum(len(t) + 1 for t in enc.encode_ordinary_batch(dev))), sealed_tokens=int(sum(len(t) + 1 for t in enc.encode_ordinary_batch(sealed))))
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
