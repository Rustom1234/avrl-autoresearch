"""SR5: which parts of a hidden law does the cheap model need to be told? (rules: loop/design/SR5_PREREG.md)

A hint cell is a mask over the terms of the law in the order of gen.LEVELS ("001" on level 3 = only the unusual
term is given). "W" gives a wrong form for the unusual term. Hint wording is that of advice.py (SR4).

    advice2.py make   --src loop/sr/problems_main --out loop/sr/problems_advice2 --only L3 --masks 100 010 101 011 111 W
    advice2.py table  --roots loop/runs/SR4 loop/runs/SR5 --probs loop/sr/problems_advice loop/sr/problems_advice2 --src loop/sr/problems_main
    advice2.py strong --attempts loop/runs/SR1/attempts --src loop/sr/problems_main --level 3
"""
import argparse, collections, glob, itertools, json, math, os, re
from pathlib import Path

import numpy as np

import advice as A
import gen

C = " In the forms below each c is a different unknown constant."
NAMES = {3: ["S", "CS", "U"], 4: ["S", "CS", "U", "US"]}


def hint_mask(mask, terms, nv):
    base = A.hint("G", terms, nv)[0]
    if mask == "W":  # wrong advice: the unusual term's slot with the next kind of the unusual list
        kinds = [n for n, _, _ in gen.UNUSUAL]
        t = next(t for t in terms if len(t) == 1 and t[0][0] == "unusual")
        j = (kinds.index(t[0][2]) + 1) % len(kinds)
        wrong = [(t[0][0], t[0][1], gen.UNUSUAL[j][0], gen.UNUSUAL[j][1], t[0][4])]
        return base + C + " One of the terms is exactly of this form: `%s`." % A.form(wrong), 0.0
    assert len(mask) == len(terms), (mask, len(terms))
    pick = [t for t, m in zip(terms, mask) if m == "1"]
    if not pick:
        return base, 0.0
    lead = ("The terms are exactly of these forms" if len(pick) == len(terms) else
            "One of the terms is exactly of this form" if len(pick) == 1 else
            "%d of the terms are exactly of these forms" % len(pick))
    return base + C + " %s: %s." % (lead, " and ".join("`%s`" % A.form(t) for t in pick)), sum(A.bits(t, nv) for t in pick)


def cmd_make(a):
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    for pdir in sorted(d for d in src.iterdir() if (d / "card.json").exists() and any(s in d.name for s in a.only)):
        card, truth = json.loads((pdir / "card.json").read_text()), json.loads((pdir / "truth.json").read_text())
        terms, nv = A.parse(truth, card)
        for mask in a.masks:
            text, b = hint_mask(mask, terms, nv)
            rung = mask if mask == "W" else "R" + mask
            d = out / ("%s__%s" % (pdir.name, rung))
            d.mkdir(parents=True, exist_ok=True)
            for f in ("train.csv", "val.csv", "test_id.csv", "test_ood.csv", "truth.json"):
                if not (d / f).exists():
                    os.symlink(pdir / f, d / f)
            (d / "card.json").write_text(json.dumps(dict(card, context=card["context"] + " " + text), indent=1))
            (d / "advice.json").write_text(json.dumps({"base": pdir.name, "rung": rung, "bits": b, "text": text,
                                                       "total_bits": sum(A.bits(t, nv) for t in terms)}, indent=1))
            print("%-22s %5.1f bits | %s" % (d.name, b, text[len(text) - 110:]))


def cell_of(rung, terms):
    """Mask of an SR4 or SR5 rung; None for rungs that are not 'these terms are given' (K)."""
    n, hard = len(terms), [any(f[0] == "unusual" for f in t) for t in terms]
    if rung.startswith("R"):
        return rung[1:]
    return {"W": "W", "G": "0" * n, "ALL": "1" * n, "U2": "".join("01"[h] for h in hard),
            "E": "".join("10"[h] for h in hard),
            "U1": "".join("1" if j == hard.index(True) else "0" for j in range(n))}.get(rung)


def load(roots, probs, src):
    """data[level][mask][problem] = list of 0/1 (chain solved within 4 steps); cost[level][mask] = USD;
    first[level][mask][problem] = list of (step, tool calls so far) at the solve, or (None, None)."""
    adv = {Path(f).parent.name: json.load(open(f)) for p in probs for f in glob.glob(p + "/*/advice.json")}
    ch = collections.defaultdict(dict)
    for root in roots:
        for f in glob.glob(root + "/trie/*/r*/**/record.json", recursive=True):
            r = json.load(open(f))
            ch[(r["problem"], r["rep"])][len(r["path"])] = r
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    cost = collections.defaultdict(lambda: collections.defaultdict(float))
    first = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    terms_of = {}
    for (p, rep), steps in ch.items():
        base = adv[p]["base"]
        if base not in terms_of:
            d = Path(src) / base
            terms_of[base] = A.parse(json.loads((d / "truth.json").read_text()), json.loads((d / "card.json").read_text()))[0]
        mask = cell_of(adv[p]["rung"], terms_of[base])
        if mask is None:
            continue
        lev = int(base.split("_")[1][1:])
        hit = [j for j in sorted(steps) if steps[j]["val"] is not None and steps[j]["val"] <= 1e-10]
        data[lev][mask][base].append(int(bool(hit) and hit[0] <= 4))
        cost[lev][mask] += sum(s["cost_usd"] for s in steps.values())
        ok = bool(hit) and hit[0] <= 4
        first[lev][mask][base].append((hit[0], sum(steps[j]["n_calls"] for j in steps if j <= hit[0])) if ok else (None, None))
    return data, cost, first


def boot(stat, probs, n=5000):
    rng, probs = np.random.default_rng(0), sorted(probs)
    draws = [stat([probs[i] for i in rng.integers(0, len(probs), len(probs))]) for _ in range(n)]
    return stat(probs), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def cmd_table(a):
    data, cost, _ = load(a.roots, a.probs, a.src)
    for lev in sorted(data):
        cells, names = data[lev], NAMES[lev]
        f = lambda m, ps=None: float(np.mean([np.mean(cells[m][p]) for p in (ps or cells[m])]))
        print("\nlevel %d   (terms in mask order: %s)" % (lev, ", ".join(names)))
        print("mask   given            chains  solved in 4  share  problems solved   USD")
        for m in sorted(cells, key=lambda m: (m == "W", m.count("1"), m)):
            v = [x for p in cells[m] for x in cells[m][p]]
            given = "wrong form for U" if m == "W" else (", ".join(n for n, b in zip(names, m) if b == "1") or "nothing")
            print("%-6s %-16s %5d   %5d        %.2f   %d of %d          %.2f" % (
                m, given, len(v), sum(v), f(m), sum(any(cells[m][p]) for p in cells[m]), len(cells[m]), cost[lev][m]))
        have = lambda *ms: all(m in cells for m in ms)
        common = lambda *ms: sorted(set.intersection(*[set(cells[m]) for m in ms]))
        if lev == 3 and have("000", "001", "010", "100", "011", "101", "110"):
            ms = ("000", "001", "010", "100", "011", "101", "110")
            def f1(ps):
                pS, pCS, pU = f("011", ps), f("101", ps), f("110", ps)
                return float(np.mean([f("100", ps) - pCS * pU, f("010", ps) - pS * pU, f("001", ps) - pS * pCS,
                                      f("000", ps) - pS * pCS * pU]))
            ps = common(*ms)
            pS, pCS, pU = f("011", ps), f("101", ps), f("110", ps)
            print("  independent-parts predictions (observed): 100 %.2f (%.2f), 010 %.2f (%.2f), 001 %.2f (%.2f), 000 %.2f (%.2f)" % (
                pCS * pU, f("100", ps), pS * pU, f("010", ps), pS * pCS, f("001", ps), pS * pCS * pU, f("000", ps)))
            print("  F1 mean observed minus predicted over %d problems: %+.3f (95%% %+.3f to %+.3f)" % ((len(ps),) + boot(f1, ps)))
            if have("111"):
                n, v = 3, lambda s: f("".join("1" if i in s else "0" for i in range(3)), ps)
                for i in range(n):
                    rest = [j for j in range(n) if j != i]
                    phi = sum(math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n) * (v(set(s) | {i}) - v(set(s)))
                              for k in range(n) for s in itertools.combinations(rest, k))
                    print("  Shapley value of being told %-2s: %+.3f" % (names[i], phi))
        if lev == 4 and have("1101", "1110"):
            ps = common("1101", "1110")
            print("  F2 f(1110) minus f(1101) over %d problems: %+.3f (95%% %+.3f to %+.3f)" % (
                (len(ps),) + boot(lambda q: f("1110", q) - f("1101", q), ps)))
            if have("1100"):
                ps = common("1101", "1110", "1100")
                print("  F3 f(1100) minus f(1101) f(1110) over %d problems: %+.3f (95%% %+.3f to %+.3f); prediction %.2f, observed %.2f" % (
                    (len(ps),) + boot(lambda q: f("1100", q) - f("1101", q) * f("1110", q), ps) + (f("1101", ps) * f("1110", ps), f("1100", ps))))


def has_kind(code, kind, v):
    """Does this candidate contain the unusual form `kind` on input v? Text patterns; free exponents and phases count."""
    s = re.sub(r"\s+", "", code.split("return", 1)[-1]).replace("np.", "")
    s = re.sub(r"\b(params|p)\[\d+\]", "c", s)
    V = r"(?<![A-Za-z_])%s(?![A-Za-z_0-9])" % re.escape(v)
    root = r"(sqrt\(%s\)|%s\*\*\(?(0\.5|c|1/2)\)?)" % (V, V)
    pats = {"sinsqrt": [r"sin\([^()]*" + root], "expsqrt": [r"exp\(-?[^()]*" + root],
            "freepow": [V + r"\*\*c", r"power\(%s,c\)" % V, r"exp\(c\*log\(%s\)\)" % V],
            "hill": [V + r"\*\*(2|c)/\([^()]*" + V + r"\*\*(2|c)\)"],
            "gauss": [r"exp\(-?[^()]*" + V + r"\*\*2", r"exp\(-?[^()]*(abs)?\([^()]*" + V + r"[^()]*\)\*\*\(?2"],
            "tanh": [r"tanh\([^()]*" + V],
            "vlogv": [V + r"\*log\(" + V + r"\)", r"log\(" + V + r"\)\*" + V],
            "logsq": [r"log(1p)?\([^()]*" + V + r"[^()]*\)\*\*2"]}
    return any(re.search(p, s) for p in pats[kind])


def cmd_strong(a):
    rows = collections.defaultdict(list)
    for pdir in sorted(Path(a.src).glob("*_L%d_s*" % a.level)):
        terms, _ = A.parse(json.loads((pdir / "truth.json").read_text()), json.loads((pdir / "card.json").read_text()))
        u = next(t for t in terms if len(t) == 1 and t[0][0] == "unusual")[0]
        for rec in sorted(Path(a.attempts).glob(pdir.name + "/*/*/record.json")):
            r, d = json.loads(rec.read_text()), rec.parent
            if not r.get("ok") or not (d / "evals.jsonl").exists():
                continue
            evs = [json.loads(l) for l in open(d / "evals.jsonl")]
            hits = [int(e["call"]) for e in evs if has_kind(e["code"], u[2], u[4])]
            solved = r.get("val") is not None and r["val"] <= 1e-10
            final = has_kind((r.get("best") or {}).get("code", ""), u[2], u[4])
            rows[r["model"]].append((pdir.name, u[2], solved, final, min(hits) if hits else None, r["n_calls"], r["cost_usd"]))
    for model, v in rows.items():
        first = [x[4] for x in v if x[4] is not None]
        print("\n%s: %d sessions, %d solved, detector fires on the final formula of %d of the %d solved" % (
            model, len(v), sum(x[2] for x in v), sum(x[2] and x[3] for x in v), sum(x[2] for x in v)))
        print("  sessions that ever send the true unusual form to the checker: %d of %d; by tool call 4: %d, 7: %d, 10: %d" % (
            len(first), len(v), sum(c <= 4 for c in first), sum(c <= 7 for c in first), sum(c <= 10 for c in first)))
        print("  among unsolved sessions: %d of %d had the form; mean cost %.2f USD, mean tool calls %.1f" % (
            sum(x[4] is not None and not x[2] for x in v), sum(not x[2] for x in v), np.mean([x[6] for x in v]), np.mean([x[5] for x in v])))
        for x in v:
            print("    %-14s %-8s solved %-5s first call with the form %s of %s" % (x[0], x[1], x[2], x[4], x[5]))

def cmd_seq(a):
    """Added after the run (not in the rules file): a stricter comparison for the hiding claims.

    The product rule lets every missing term have the whole budget. Here the missing terms are found one after
    another and the efforts add: effort(M) = base + sum over missing terms of (effort with only that term missing
    - base). The prediction is the share of chain combinations (same problem) whose summed effort fits the budget."""
    data, _, first = load(a.roots, a.probs, a.src)
    for lev in sorted(data):
        n, cells = len(NAMES[lev]), data[lev]
        full = "1" * n
        single = [full[:i] + "0" + full[i + 1:] for i in range(n)]
        if not all(m in cells for m in single + [full]):
            continue
        f = lambda m, ps: float(np.mean([np.mean(cells[m][p]) for p in ps]))
        base = [float(np.mean([x[k] for v in first[lev][full].values() for x in v])) for k in (0, 1)]
        def pred(m, ps, k, budget):
            out = []
            for p in ps:
                combos = list(itertools.product(*[first[lev][single[i]][p] for i in range(n) if m[i] == "0"]))
                if budget is None:  # every term has the whole budget (product rule, per problem)
                    out.append(np.mean([all(x[0] is not None for x in c) for c in combos]))
                else:
                    out.append(np.mean([all(x[0] is not None for x in c) and
                                        base[k] + sum(max(x[k] - base[k], 0) for x in c) <= budget for c in combos]))
            return float(np.mean(out))
        everywhere = sorted(set.intersection(*[set(cells[m]) for m in single]))
        multi = sorted((m for m in cells if m != "W" and m.count("0") >= 2), key=lambda m: (m.count("0"), m))
        print("\nlevel %d: all terms given solves at step %.2f after %.1f tool calls on average" % (lev, base[0], base[1]))
        print("missing          problems  observed  product  efforts add (steps)  efforts add (tool calls)")
        for m in multi:
            ps = sorted(set(cells[m]) & set(everywhere))
            print("%-16s %5d     %.2f      %.2f     %.2f                 %.2f" % (
                ", ".join(x for x, b in zip(NAMES[lev], m) if b == "0"), len(ps), f(m, ps), pred(m, ps, 0, None),
                pred(m, ps, 0, 4), pred(m, ps, 1, 56)))
        big = [m for m in multi if set(cells[m]) >= set(everywhere)]
        for name, k, budget in (("product", 0, None), ("efforts add, steps", 0, 4), ("efforts add, tool calls", 1, 56)):
            stat = lambda ps: float(np.mean([f(m, ps) - pred(m, ps, k, budget) for m in big]))
            print("  observed minus predicted (%s), mean over cells %s, %d problems: %+.3f (95%% %+.3f to %+.3f)" % (
                (name, " ".join(big), len(everywhere)) + boot(stat, everywhere)))


def cmd_wrong(a):
    """Added after the run: in the wrong-hint cell, does the chain keep the wrong form, and does it ever try the true one?"""
    kinds = [k for k, _, _ in gen.UNUSUAL]
    rows = []
    for d in sorted(Path(a.root).glob("trie/*__W/r*")):
        base = d.parent.name.split("__")[0]
        src = Path(a.src) / base
        terms, _ = A.parse(json.loads((src / "truth.json").read_text()), json.loads((src / "card.json").read_text()))
        u = next(t for t in terms if len(t) == 1 and t[0][0] == "unusual")[0]
        wrong = kinds[(kinds.index(u[2]) + 1) % len(kinds)]
        recs = sorted((json.loads(f.read_text()) for f in d.rglob("record.json")), key=lambda r: len(r["path"]))
        evs = [json.loads(l)["code"] for f in d.rglob("evals.jsonl") for l in open(f)]
        last = (recs[-1].get("best") or {}).get("code", "")
        rows.append((base, u[2], wrong, has_kind(last, wrong, u[4]), any(has_kind(c, u[2], u[4]) for c in evs),
                     np.mean([has_kind(c, wrong, u[4]) for c in evs]), len(evs), recs[-1]["val"]))
    print("wrong-hint chains: %d; final formula keeps the wrong form in %d; ever sent the true form to the checker in %d" % (
        len(rows), sum(r[3] for r in rows), sum(r[4] for r in rows)))
    print("share of all checked formulas that contain the wrong form: %.2f" % np.mean([r[5] for r in rows]))
    for r in rows:
        print("  %-14s true %-8s told %-8s final keeps wrong %-5s tried true %-5s checked %3d  final val %.1e" % (r[:5] + r[6:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="+", required=True)
    p.add_argument("--masks", nargs="+", required=True)
    p = sub.add_parser("table")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--probs", nargs="+", required=True)
    p.add_argument("--src", required=True)
    p = sub.add_parser("seq")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--probs", nargs="+", required=True)
    p.add_argument("--src", required=True)
    p = sub.add_parser("wrong")
    p.add_argument("--root", required=True)
    p.add_argument("--src", required=True)
    p = sub.add_parser("strong")
    p.add_argument("--attempts", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--level", type=int, default=3)
    a = ap.parse_args()
    {"make": cmd_make, "table": cmd_table, "strong": cmd_strong, "seq": cmd_seq, "wrong": cmd_wrong}[a.cmd](a)
