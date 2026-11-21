"""SR3: does an "out of ideas" signal from the cheap model's own proposals predict whether a chain will solve?

Stored sessions only (no model calls). Rules: loop/design/SR3_PREREG.md.
    novelty_test.py --main loop/runs/SR2 --l4 loop/runs/SR2L4 --cards loop/sr/problems_main
"""
import argparse, collections, glob, json, math, os, re
import numpy as np

TAU = 1e-10
MATH = {"exp", "log", "log1p", "sqrt", "sin", "cos", "tan", "tanh", "sinh", "cosh", "arctan", "arcsinh", "abs",
        "cbrt", "expm1", "log10", "log2", "power", "sign", "maximum", "minimum", "square", "reciprocal", "arctan2"}
WORD = re.compile(r"[A-Za-z_]\w*")


def blocks(code, inputs):
    """Building blocks used in one piece of tool-call code."""
    out = set()
    for m in re.finditer(r"np\.(\w+)\s*\(", code):
        if m.group(1) not in MATH:
            continue
        i, depth = m.end(), 1
        while i < len(code) and depth:
            depth += (code[i] == "(") - (code[i] == ")")
            i += 1
        arg = code[m.end():i - 1]
        vs = sorted({w for w in WORD.findall(arg) if w in inputs})
        inner = sorted({f for f in re.findall(r"np\.(\w+)\s*\(", arg) if f in MATH})
        if vs or inner:
            out.add("%s(%s|%s)" % (m.group(1), ",".join(inner), ",".join(vs)))
    for m in re.finditer(r"/\s*\(([^()]*)\)", code):
        vs = sorted({w for w in WORD.findall(m.group(1)) if w in inputs})
        if vs:
            out.add("ratio(%s)" % ",".join(vs))
    for m in re.finditer(r"/\s*([A-Za-z_]\w*)", code):
        if m.group(1) in inputs:
            out.add("inv(%s)" % m.group(1))
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\*\*\s*\(?(-?[\w\.]+)", code):
        if m.group(1) in inputs:
            try:
                v = float(m.group(2))
                cls = "int%d" % v if v == int(v) else "frac"
            except ValueError:
                cls = "sym"
            out.add("pow(%s,%s)" % (m.group(1), cls))
    return out


def find_err(o):
    """Training error in the reply of the evaluate tool (first key that names a training nmse)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if "train" in k and isinstance(v, (int, float)):
                return float(v)
        for v in o.values():
            r = find_err(v)
            if r is not None:
                return r
    return None


def load(root, cards):
    """(problem, rep) -> list of steps; a step = dict(solved, best, blocks, funcs)."""
    chains = collections.defaultdict(dict)
    for f in glob.glob(root + "/trie/*/r*/**/record.json", recursive=True):
        r = json.load(open(f))
        if any(m != "haiku" for m in r["path"]):
            continue
        inputs = {v["name"] for v in json.load(open(os.path.join(cards, r["problem"], "card.json")))["inputs"]}
        bl, fu = set(), []
        cf = os.path.join(os.path.dirname(f), "work", "calls.jsonl")
        if os.path.exists(cf):
            for line in open(cf):
                c = json.loads(line)
                bl |= blocks(c.get("code") or "", inputs)
                if c.get("tool") == "evaluate":
                    try:
                        e = find_err(json.loads(c["out"]))
                    except (ValueError, TypeError):
                        e = None
                    if e is not None and e > 0:
                        fu.append(round(math.log10(e), 2))
        chains[(r["problem"], r["rep"])][len(r["path"])] = dict(
            solved=r["val"] is not None and r["val"] <= TAU,
            best=r["best"]["train_nmse"] if r.get("best") else None, blocks=bl, funcs=fu)
    return {k: [s[j] for j in sorted(s)] for k, s in chains.items()}


def signals(steps, j, win=2):
    """Signals from steps 1..j of one chain."""
    errs = [s["best"] for s in steps[:j] if s["best"]]
    seen, first = set(), {}
    for i, s in enumerate(steps[:j]):
        for b in s["blocks"] - seen:
            first[b] = i
        seen |= s["blocks"]
    funcs = [x for s in steps[:j] for x in s["funcs"]]
    cnt = collections.Counter(funcs)
    gt = sum(1 for v in cnt.values() if v == 1) / len(funcs) if funcs else 1.0
    return dict(err=math.log10(min(errs)) if errs else 0.0, gt=gt, breadth=float(len(seen)),
                recent=float(sum(1 for i in first.values() if i >= j - win)), nfun=float(len(funcs)))


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    for v in np.unique(score):                      # mid-ranks for ties
        r[score == v] = r[score == v].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def logit_lopo(X, y, groups, ridge=1.0):
    """Leave-one-problem-out predictions of a ridge logistic model."""
    X, y, pred = np.asarray(X, float), np.asarray(y, float), np.zeros(len(y))
    for g in set(groups):
        te = np.array([h == g for h in groups])
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
        A = np.column_stack([np.ones((~te).sum()), (X[~te] - mu) / sd])
        w = np.zeros(A.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-A @ w))
            H = A.T @ (A * (p * (1 - p))[:, None]) + ridge * np.eye(len(w))
            w += np.linalg.solve(H, A.T @ (y[~te] - p) - ridge * w)
        pred[te] = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ w
    return pred


def level(p):
    return p.split("_")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--l4", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    C = load(a.main, a.cards)
    L4 = {k: v for k, v in load(a.l4, a.cards).items() if level(k[0]) == "L4"}
    print("chains: main %d, level 4 %d; evaluated candidates per step (main): mean %.2f" % (
        len(C), len(L4), np.mean([len(s["funcs"]) for v in C.values() for s in v])))

    # E1: chains of levels 2, 6, 3 unsolved after 4 steps; outcome = solved by step 12
    rows = []
    for (p, r), st in sorted(C.items()):
        if level(p) in ("L2", "L6", "L3") and len(st) >= 4 and not any(s["solved"] for s in st[:4]):
            rows.append((p, signals(st, 4), any(s["solved"] for s in st[4:12])))
    y, g = np.array([r[2] for r in rows]), [r[0] for r in rows]
    names = ["err", "gt", "breadth", "recent"]
    X = np.array([[r[1][n] for n in names] for r in rows])
    print("\nE1: %d chains unsolved after 4 steps, %d solved by step 12, %d problems" % (len(y), y.sum(), len(set(g))))
    for i, n in enumerate(names):
        print("  AUC of %-8s alone: %.3f (below 0.5 means lower values go with solving)" % (n, auc(X[:, i], y)))
    p0, p1 = logit_lopo(X[:, :1], y, g), logit_lopo(X, y, g)
    d = auc(p1, y) - auc(p0, y)
    rng, probs, ds = np.random.default_rng(0), sorted(set(g)), []
    idx = {q: [i for i, h in enumerate(g) if h == q] for q in probs}
    for _ in range(a.boot):
        ii = np.concatenate([idx[q] for q in rng.choice(probs, len(probs))])
        v = auc(p1[ii], y[ii]) - auc(p0[ii], y[ii])
        if not math.isnan(v):
            ds.append(v)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print("  leave-one-problem-out AUC: error alone %.3f, error + three signals %.3f, difference %+.3f (95%% %+.3f to %+.3f)" % (
        auc(p0, y), auc(p1, y), d, lo, hi))
    print("  E1 %s" % ("PASS" if d >= 0.05 and lo > 0 else "FAIL"))

    # E2: the bound at every chain step of every level (main set and level 4)
    B = []
    for st in list(C.values()) + list(L4.values()):
        for j in range(1, len(st)):
            if not any(s["solved"] for s in st[:j]):
                B.append((signals(st, j)["gt"], st[j]["solved"]))
    b, s = np.array([x[0] for x in B]), np.array([x[1] for x in B])
    print("\nE2: %d chain steps; Good-Turing bound: median %.2f, share below 0.5: %.2f" % (len(b), np.median(b), (b < 0.5).mean()))
    ok = True
    for lo_, hi_ in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        m = (b >= lo_) & (b < hi_)
        if m.any():
            print("  bound in [%.2f, %.2f): %4d steps, mean bound %.2f, next step solved %.3f" % (lo_, hi_, m.sum(), b[m].mean(), s[m].mean()))
            ok &= s[m].mean() <= b[m].mean()
    print("  E2 %s" % ("PASS" if ok and (b < 0.5).mean() >= 0.5 else "FAIL"))

    # E3: level 4 at step 12, outcome = solved by step 24 (descriptive)
    rows = [(signals(st, 12), any(s["solved"] for s in st[12:24])) for st in L4.values() if len(st) >= 12]
    y4 = np.array([r[1] for r in rows])
    print("\nE3: level 4, %d chains at step 12, %d solved by step 24" % (len(y4), y4.sum()))
    for n in names:
        print("  AUC of %-8s: %.3f" % (n, auc([r[0][n] for r in rows], y4)))


if __name__ == "__main__":
    main()
