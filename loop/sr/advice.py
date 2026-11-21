"""SR4: how many bits of advice does the cheap model need? (rules: loop/design/SR4_PREREG.md)

A hidden law is a sum of terms; a term is a constant times one or two factors drawn from known pools (gen.py). So
the information in a hint about the structure can be counted exactly in bits under the generator's own odds.

    advice.py make  --src loop/sr/problems_main --out loop/sr/problems_advice --only L4_s1 --rungs G K U1 U2 E ALL
    advice.py table --root loop/runs/SR4 --probs loop/sr/problems_advice

`make` writes one problem folder per (problem, rung): the same data, with the hint added to the problem text.
The solver code is untouched; the chains are run with trie.py as before.
"""
import argparse, collections, glob, json, math, os, re
from pathlib import Path

import gen

NUM = r"[-+0-9.e]+"
RUNGS = ("G", "K", "U1", "U2", "E", "ALL")


def split_top(expr):
    """Split a sum at ' + ' outside brackets."""
    out, depth, start, i = [], 0, 0, 0
    while i < len(expr):
        depth += (expr[i] == "(") - (expr[i] == ")")
        if depth == 0 and expr.startswith(" + ", i):
            out.append(expr[start:i])
            start = i + 3
            i += 2
        i += 1
    return out + [expr[start:]]


def parse(truth, card):
    """The terms of a law as lists of (pool name, factor name, template, input), in the order of gen.LEVELS."""
    names = [v["name"] for v in card["inputs"]]
    pools_of = gen.LEVELS[truth["level"]]
    pieces = split_top(truth["expr"])
    assert len(pieces) == len(pools_of), truth["expr"]
    terms = []
    for piece, pools in zip(pieces, pools_of):
        body = piece.split("*", 1)[1]
        parts = body[1:-1].split(")*(") if len(pools) > 1 else [body]
        term = []
        for part, pool in zip(parts, pools):
            hit = [(n, tpl, v) for n, tpl, _ in pool for v in names
                   if re.fullmatch(re.escape(tpl).replace(r"\{v\}", v).replace(r"\{c0\}", NUM), part)]
            assert len(hit) == 1, (part, hit)
            pname = {id(gen.SIMPLE): "simple", id(gen.COMMON): "common", id(gen.UNUSUAL): "unusual"}[id(pool)]
            term.append((pname, len(pool)) + hit[0])
        terms.append(term)
    return terms, len(names)


def form(term, var=None):
    return "c*" + "*".join("(%s)" % tpl.format(v=var or v, c0="c") for _, _, _, tpl, v in term)


def bits(term, nv):
    """Bits needed to name this term's structure under the generator's odds (factor kinds and inputs)."""
    k = len(term)
    return sum(math.log2(size) for _, size, _, _, _ in term) + math.log2(math.perm(nv, k))


def hint(rung, terms, nv):
    """(text, bits) of one rung. Unusual terms are the ones that contain a factor from the unusual pool."""
    hard = [t for t in terms if any(f[0] == "unusual" for f in t)]
    easy = [t for t in terms if t not in hard]
    base = ("Reliable information from a colleague who knows this system (it is correct): the law is a sum of %d "
            "terms. Each term is a constant times one factor, or a constant times a product of two factors that "
            "depend on different inputs." % len(terms))
    c = " In the forms below each c is a different unknown constant."
    if rung == "G":
        return base, 0.0
    if rung == "K":
        kinds = [f for t in hard for f in t if f[0] == "unusual"]
        txt = " and ".join("`%s`" % f[3].format(v="u", c0="c") for f in kinds)
        same = len({f[2] for f in kinds}) < len(kinds)
        b = sum(math.log2(f[1]) for f in kinds) - (0 if same or len(kinds) < 2 else math.log2(math.factorial(len(kinds))))
        return base + c + " Among the factors are these kinds, where u stands for one of the inputs: %s." % txt, b
    pick = {"U1": hard[:1], "U2": hard, "E": easy, "ALL": terms}[rung]
    lead = "The terms are exactly of these forms" if rung == "ALL" else (
        "One of the terms is exactly of this form" if len(pick) == 1 else "%d of the terms are exactly of these forms" % len(pick))
    return base + c + " %s: %s." % (lead, " and ".join("`%s`" % form(t) for t in pick)), sum(bits(t, nv) for t in pick)


def cmd_make(a):
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    for pdir in sorted(d for d in src.iterdir() if (d / "card.json").exists() and any(s in d.name for s in a.only)):
        card, truth = json.loads((pdir / "card.json").read_text()), json.loads((pdir / "truth.json").read_text())
        terms, nv = parse(truth, card)
        for rung in a.rungs:
            text, b = hint(rung, terms, nv)
            d = out / ("%s__%s" % (pdir.name, rung))
            d.mkdir(parents=True, exist_ok=True)
            for f in ("train.csv", "val.csv", "test_id.csv", "test_ood.csv", "truth.json"):
                if not (d / f).exists():
                    os.symlink(pdir / f, d / f)
            (d / "card.json").write_text(json.dumps(dict(card, context=card["context"] + " " + text), indent=1))
            (d / "advice.json").write_text(json.dumps({"base": pdir.name, "rung": rung, "bits": b, "text": text,
                                                       "total_bits": sum(bits(t, nv) for t in terms)}, indent=1))
            print("%-22s %5.1f of %.1f bits | %s" % (d.name, b, sum(bits(t, nv) for t in terms), text[len(text) - 150:]))


def cmd_table(a):
    adv = {Path(f).parent.name: json.load(open(f)) for f in glob.glob(a.probs + "/*/advice.json")}
    ch = collections.defaultdict(dict)
    for f in glob.glob(a.root + "/trie/*/r*/**/record.json", recursive=True):
        r = json.load(open(f))
        ch[(r["problem"], r["rep"])][len(r["path"])] = r
    import numpy as np
    by = collections.defaultdict(list)
    for (p, rep), steps in ch.items():
        hit = [j for j in sorted(steps) if steps[j]["val"] is not None and steps[j]["val"] <= 1e-10]
        lev = adv[p]["base"].split("_")[1]
        by[(lev, adv[p]["rung"])].append((adv[p]["base"], hit[0] if hit else None, adv[p]["bits"],
                                          sum(s["cost_usd"] for s in steps.values())))
    print("level rung  bits  chains  solved within 1, 2, 4 steps   problems with a solve   USD")
    for lev in sorted({k[0] for k in by}):
        for rung in RUNGS:
            v = by.get((lev, rung), [])
            if not v:
                continue
            s = [sum(1 for _, t, *_ in v if t and t <= k) for k in (1, 2, 4)]
            print("%-5s %-4s %5.1f  %5d   %2d, %2d, %2d                    %d of %d            %.2f" % (
                lev, rung, sum(x[2] for x in v) / len(v), len(v), *s, len({p for p, t, *_ in v if t}),
                len({p for p, *_ in v}), sum(x[3] for x in v)))
        # claim S1: per problem, share of chains solved within 4 steps under U2 minus under E
        share = lambda rung: {p: np.mean([bool(t and t <= 4) for q, t, *_ in by[(lev, rung)] if q == p])
                              for p in {q for q, *_ in by.get((lev, rung), [])}}
        u, e = share("U2"), share("E")
        common = sorted(set(u) & set(e))
        if common:
            d = np.array([u[p] - e[p] for p in common])
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            print("  %s: U2 minus E over %d problems: %+.3f (95%% %+.3f to %+.3f); problems where U2 is ahead %d, behind %d" % (
                lev, len(d), d.mean(), *np.percentile(bs, [2.5, 97.5]), (d > 0).sum(), (d < 0).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="+", required=True)
    p.add_argument("--rungs", nargs="+", default=list(RUNGS))
    p = sub.add_parser("table")
    p.add_argument("--root", required=True)
    p.add_argument("--probs", required=True)
    a = ap.parse_args()
    {"make": cmd_make, "table": cmd_table}[a.cmd](a)
