"""Pool v3: clean target text is scarce and most of the pool is graded, so what to keep is an empirical question.

    python build_pool_v3.py --seed 0      -> data/curation/pool_v3/

Same wiki split as v1 and v2 for the same seed (same target sample, dev and sealed sets). Amounts are in units of T.
Wiki-derived kinds use disjoint source articles:
  clean_wiki     0.15  untouched
  wiki_detok     0.30  same text with WikiText spacing removed ("word , word" -> "word, word", no "= Title =" marks)
  wiki_typo      0.30  3 percent of letters deleted, swapped or replaced
  wiki_partshuf  0.30  30 percent of sentences have their words shuffled
  dmg_wordshuf   0.30  every sentence shuffled (v2's hard junk)
  wiki_markov    0.30  text sampled from a word trigram model of wiki: locally fluent, no meaning
plus clean_web 0.30, dmg_spam 0.25, code 0.15, qa 0.10, exact and near duplicates of clean wiki 0.05 each, and
boilerplate, base64 and letter-shuffled web (0.10 together).
"""
import argparse, json, random, re
from collections import defaultdict
import numpy as np
import tiktoken
from common import DATA, T
import build_pool as B

AMOUNT = dict(clean_wiki=0.15, wiki_detok=0.30, wiki_typo=0.30, wiki_partshuf=0.30, dmg_wordshuf=0.30, wiki_markov=0.30,
              clean_web=0.30, dmg_spam=0.25, code=0.15, qa=0.10, dup_exact_wiki=0.05, dup_near_wiki=0.05,
              dmg_boiler=0.04, dmg_b64=0.03, dmg_charshuf=0.03)
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def detok(s):
    """WikiText -> ordinary prose spacing."""
    s = re.sub(r"^(= )+(.*?)( =)+$", r"\2", s, flags=re.M)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    return s


def typo(s, rng, rate=0.03):
    out = []
    for ln in s.split("\n"):
        if ln.startswith("="):
            out.append(ln)
            continue
        c = list(ln)
        for i in range(len(c) - 1):
            if c[i].isalpha() and rng.random() < rate:
                r = rng.random()
                if r < 0.34:
                    c[i] = ""
                elif r < 0.67:
                    c[i] = rng.choice(LETTERS)
                elif c[i + 1].isalpha():
                    c[i], c[i + 1] = c[i + 1], c[i]
        out.append("".join(c))
    return "\n".join(out)


def part_shuffle(s, rng, p=0.3):
    lines = []
    for ln in s.split("\n"):
        if ln.startswith("="):
            lines.append(ln)
            continue
        sents = []
        for sent in ln.split(" . "):
            w = sent.split(" ")
            if rng.random() < p:
                rng.shuffle(w)
            sents.append(" ".join(w))
        lines.append(" . ".join(sents))
    return "\n".join(lines)


class Markov:
    """Word trigram model with one step in four taken from the bigram table, so that it does not copy long spans."""
    def __init__(self, texts):
        self.tri, self.bi, self.starts = defaultdict(list), defaultdict(list), []
        for t in texts:
            for ln in t.split("\n"):
                w = ln.split(" ")
                if ln.startswith("=") or len(w) < 8:
                    continue
                self.starts.append((w[0], w[1]))
                for a, b, c in zip(w, w[1:], w[2:]):
                    self.tri[(a, b)].append(c)
                    self.bi[b].append(c)

    def paragraph(self, rng, n_words):
        a, b = rng.choice(self.starts)
        w = [a, b]
        while len(w) < n_words:
            nxt = self.tri.get((a, b)) if rng.random() < 0.75 else self.bi.get(b)
            if not nxt:
                a, b = rng.choice(self.starts)
                w += [".", a, b]
                continue
            c = rng.choice(nxt)
            w.append(c)
            a, b = b, c
        return " ".join(w)

    def doc(self, rng, title):
        target, out, n = rng.randint(2500, 8000), ["= %s =" % title], 0
        while n < target:
            p = self.paragraph(rng, rng.randint(40, 160))
            out.append(p)
            n += len(p)
        return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    enc = tiktoken.get_encoding("gpt2")
    ntok = lambda d: len(enc.encode_ordinary(d))

    wiki = B.read_wiki(); rng.shuffle(wiki)                       # same first use of rng as build_pool.py: same split
    n0 = B.N_TARGET + B.N_DEV + B.N_SEALED
    target_sample, dev, sealed = wiki[:B.N_TARGET], wiki[B.N_TARGET:B.N_TARGET + B.N_DEV], wiki[B.N_TARGET + B.N_DEV:n0]
    rest = wiki[n0:]
    docs, labels = [], []

    def add(d, kind, src=None):
        docs.append(d); labels.append(dict(kind=kind, src=src))
        return ntok(d)

    def take(pool, kind, fn=lambda s: s):
        tot, ids = 0, []
        while tot < AMOUNT[kind] * T:
            ids.append(len(docs))
            tot += add(B.trunc(fn(pool.pop(0))), kind)
        return ids

    wiki_ids = take(rest, "clean_wiki")
    take(rest, "wiki_detok", detok)
    take(rest, "wiki_typo", lambda s: typo(s, rng))
    take(rest, "wiki_partshuf", lambda s: part_shuffle(s, rng))
    take(rest, "dmg_wordshuf", lambda s: B.word_shuffle(s, rng))
    mk_src = [rest.pop(0) for _ in range(3000)]                  # articles used only to fit the trigram model
    mk = Markov(mk_src)
    titles = [re.match(r"= (.*) =", s).group(1) for s in mk_src if re.match(r"= (.*) =", s)]
    tot = 0
    while tot < AMOUNT["wiki_markov"] * T:
        tot += add(mk.doc(rng, rng.choice(titles)), "wiki_markov")
    web_src, code_src, qa_src = B.read_web(rng, n_groups=18), B.read_code(rng), B.read_qa(rng)
    take(web_src, "clean_web"); take(code_src, "code"); take(qa_src, "qa")
    for kind, fn in (("dup_exact_wiki", lambda s: s), ("dup_near_wiki", lambda s: B.near_dup(s, rng))):
        tot = 0
        for s_id in rng.sample(wiki_ids, len(wiki_ids)):
            if tot >= AMOUNT[kind] * T:
                break
            tot += add(fn(docs[s_id]), kind, s_id)
    take(web_src, "dmg_charshuf", lambda s: B.char_shuffle(s, rng))
    for kind, mkdoc in (("dmg_boiler", lambda: B.boiler_doc(web_src[:50], rng)), ("dmg_b64", lambda: B.b64_doc(rng))):
        tot = 0
        while tot < AMOUNT[kind] * T:
            tot += add(mkdoc(), kind)
    words = re.findall(r"[a-z]{4,}", " ".join(rest[:300]).lower())
    tot = 0
    while tot < AMOUNT["dmg_spam"] * T:
        tot += add(B.spam_doc(words, rng), "dmg_spam")

    perm = list(range(len(docs))); rng.shuffle(perm)
    inv = {old: new for new, old in enumerate(perm)}
    docs2 = [docs[o] for o in perm]
    labels2 = [dict(kind=labels[o]["kind"], src=None if labels[o]["src"] is None else inv[labels[o]["src"]]) for o in perm]
    out = DATA / "pool_v3"; out.mkdir(parents=True, exist_ok=True)
    json.dump(docs2, open(out / "docs.json", "w"))
    json.dump(labels2, open(out / "labels_HIDDEN.json", "w"))
    json.dump(target_sample, open(out / "target_sample.json", "w"))
    toks = enc.encode_ordinary_batch(docs2, num_threads=16)
    lens = np.array([len(t) + 1 for t in toks], dtype=np.int64)
    np.save(out / "pool_tokens.npy", np.concatenate([np.array(t + [enc.eot_token], dtype=np.uint16) for t in toks]))
    np.save(out / "pool_offsets.npy", np.concatenate([[0], np.cumsum(lens)]))
    for name, ds in (("dev", dev), ("sealed", sealed)):
        tt = enc.encode_ordinary_batch(ds, num_threads=16)
        np.save(out / f"{name}_tokens.npy", np.concatenate([np.array(t + [enc.eot_token], dtype=np.uint16) for t in tt]))
    kinds = defaultdict(int)
    for l, n in zip(labels2, lens):
        kinds[l["kind"]] += int(n)
    summ = dict(preset="v3", seed=args.seed, n_docs=len(docs2), pool_tokens=int(lens.sum()), pool_over_T=float(lens.sum() / T),
                tokens_by_kind_over_T={k: round(v / T, 4) for k, v in sorted(kinds.items())})
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
