"""SR6, post hoc: what did the chains do with the term check? Reads stored sessions only (no model calls).

    iso_why.py --tool ../runs/SR6 --base ../runs/SR4 ../runs/SR5 --src problems_main [--show bio_L3_s2__G/r0]

"Proposed the unusual form" means: some scored law has a term that depends only on the unusual term's input and
contains the unusual form (text patterns of advice2.has_kind, after constants and simple assignments are
written into the return line). A solved chain counts as proposed.
"""
import argparse, ast, collections, copy, json
from pathlib import Path

import advice as A
import advice2 as A2

COMMON = ("freepow", "tanh", "gauss", "vlogv")          # split made after seeing the data, see RESULT_SR6.md
RARE = ("logsq", "sinsqrt", "expsqrt")


class Inline(ast.NodeTransformer):
    def __init__(self, env):
        self.env = env

    def visit_Name(self, node):
        return copy.deepcopy(self.env[node.id]) if isinstance(node.ctx, ast.Load) and node.id in self.env else node


def flat(code):
    """The law as one expression: names bound to constants become params[0], other simple assignments are inlined."""
    try:
        fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
    except (SyntaxError, StopIteration):
        return None
    env, const = {}, ast.parse("params[0]", mode="eval").body
    for st in fn.body:
        if isinstance(st, ast.Assign):
            used = {x.id for x in ast.walk(st.value) if isinstance(x, ast.Name)}
            if used <= {"params", "p"}:
                for t in st.targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            env[x.id] = const
            elif len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                env[st.targets[0].id] = Inline(env).visit(copy.deepcopy(st.value))
        elif isinstance(st, ast.Return) and st.value is not None:
            return Inline(env).visit(copy.deepcopy(st.value))
    return None


def tried(code, kind, v, names):
    """Does the law have a term in v alone that contains the form `kind`?"""
    tree = flat(code)
    if tree is None:
        return A2.has_kind(code, kind, v)
    terms = []

    def walk(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            walk(n.left), walk(n.right)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            walk(n.operand)
        else:
            terms.append(n)

    walk(tree)
    return any({x.id for x in ast.walk(t) if isinstance(x, ast.Name)} & set(names) == {v}
               and A2.has_kind("return " + ast.unparse(t), kind, v) for t in terms)


def chain(d, src):
    """One chain folder -> dict(solved, s, u, names, codes, checks); s and u are the simple and unusual factors."""
    pdir = Path(src) / d.parent.name.split("__")[0]
    card = json.loads((pdir / "card.json").read_text())
    terms, _ = A.parse(json.loads((pdir / "truth.json").read_text()), card)
    u = next(t for t in terms if len(t) == 1 and t[0][0] == "unusual")[0]
    s = next(t for t in terms if len(t) == 1 and t[0][0] == "simple")[0]
    names = [v["name"] for v in card["inputs"]]
    recs = sorted(d.rglob("record.json"), key=lambda f: len(f.parts))
    solved = any((json.loads(f.read_text()).get("val") or 1) <= 1e-10 for f in recs)
    codes, checks = [], []
    for f in recs:
        w, step = f.parent / "work", len(f.parts) - len(d.parts) - 1
        ev = {}
        if (w / "evals.jsonl").exists():
            for l in open(w / "evals.jsonl"):
                e = json.loads(l)
                ev[e["call"]] = e["code"]
                codes.append(e["code"])
        if (w / "parts.jsonl").exists():
            for l in open(w / "parts.jsonl"):
                p = json.loads(l)
                code = ev.get(p["call"], "")
                checks.append(dict(p, step=step, code=code, u=bool(code) and tried(code, u[2], u[4], names)))
    return dict(solved=solved, s=s, u=u, names=names, codes=codes, checks=checks, name="%s %s" % (d.parent.name, d.name),
                prop=solved or any(tried(c, u[2], u[4], names) for c in codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--show")
    a = ap.parse_args()
    if a.show:
        c = chain(Path(a.tool) / "trie" / a.show, a.src)
        print("truth: simple %s on %s, unusual %s on %s; solved %s" % (c["s"][2], c["s"][4], c["u"][2], c["u"][4], c["solved"]))
        for p in c["checks"]:
            print("step %d call %2d nmse %.1e mixing %-5s single %s\n    %s" % (
                p["step"], p["call"], p["train_nmse"], p.get("mixing"), p.get("single"), p["code"].split("return", 1)[-1].strip()[:230]))
        return

    arms = [("check, nothing told", a.tool, "__G"), ("check, S told", a.tool, "__R100")]
    for root in a.base:
        arms += [("no check, %s" % n, root, suf) for n, suf in (("nothing told", "__G"), ("S told", "__R100"), ("S and CS told", "__E"))]
    print("proposed / solved of chains, by kind of the unusual form")
    print("arm".ljust(26) + "".join(k.rjust(12) for k in COMMON + RARE) + "   | common forms | rare forms")
    for name, root, suf in arms:
        rows = [chain(d, a.src) for d in sorted(Path(root).glob("trie/*_L3_s*%s/r*" % suf))]
        if not rows:
            continue
        cell = lambda ks: "%d/%d of %d" % (sum(r["prop"] for r in rows if r["u"][2] in ks), sum(r["solved"] for r in rows if r["u"][2] in ks),
                                           sum(r["u"][2] in ks for r in rows))
        print(name.ljust(26) + "".join(cell((k,)).rjust(12) for k in COMMON + RARE) + "   | %s | %s" % (cell(COMMON).rjust(12), cell(RARE).rjust(10)))

    for cell, suf in (("000", "__G"), ("100", "__R100")):
        st, first_at, kept, after, lines = collections.Counter(), [], 0, 0, []
        for d in sorted(Path(a.tool).glob("trie/*%s/r*" % suf)):
            c = chain(d, a.src)
            checks, s, u = c["checks"], c["s"], c["u"]
            st["chains"] += 1
            st["term checks"] += len(checks)
            hit = [i for i, p in enumerate(checks) if p.get("mixing")]
            if not hit:
                continue
            i0, later = hit[0], checks[hit[0] + 1:]
            first_at.append(checks[i0]["step"])
            after += len(later)
            kept += sum(bool(p.get("mixing")) for p in later)
            st["chains told 'mixing part RIGHT' at least once"] += 1
            st["  of these solved"] += c["solved"]
            st["  of these lost the mixing part in a later check"] += any(not p.get("mixing") for p in later)
            st["  of these told the group of the unusual term's input is RIGHT"] += any(p.get("mixing") and (p.get("single") or {}).get(u[4]) for p in checks)
            if s[4] != u[4]:
                both = [i for i, p in enumerate(checks) if p.get("mixing") and (p.get("single") or {}).get(s[4])]
                if both:                            # the chain now knows: mixing right, simple right, only the unusual input's group is wrong
                    st["chains told 'mixing RIGHT' and 'simple term's group RIGHT' (only the unusual term left)"] += 1
                    st["  of these solved "] += c["solved"]
                    lines.append("    %-24s %-8s at step %d; later checks %2d, later sessions %d; proposed the unusual form later: %-5s solved: %s" % (
                        c["name"], u[2], checks[both[0]]["step"], len(checks) - both[0] - 1, 4 - checks[both[0]]["step"],
                        c["solved"] or any(p["u"] for p in checks[both[0]:]), c["solved"]))
        print("\ncell %s" % cell)
        for k, v in st.items():
            print("  %-92s %d" % (k, v))
        print("  first 'mixing RIGHT' came at step (1 to 4): %s" % sorted(first_at))
        print("  later checks after the first 'mixing RIGHT': %d, of which the mixing part was still RIGHT: %d" % (after, kept))
        print("  chains with only the unusual term left:\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
