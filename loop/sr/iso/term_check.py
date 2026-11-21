#!/usr/bin/env python3
"""Check the terms of a candidate law one group at a time (SR6). Prints one JSON object.

    term_check.py --code cand.py --train train.csv

The return expression of equation() is split at its top-level + and - into terms.
1. The terms that mix two or more inputs are scored together, with free curves of every input (fit_part.py).
2. If they are right, then for each input v: the mixing terms plus the terms of v alone are scored with free curves
   of the other inputs. Terms of the same input cannot be told apart by the curves, so they get one verdict.
Each part check is its own one-thread process under the limits of the tool server.
"""
import argparse, ast, copy, json, os, subprocess, sys, tempfile
from pathlib import Path

ISO = Path(__file__).resolve().parent
PY = str(ISO.parent / ".venv/bin/python")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
LIMIT = ["prlimit", "--as=%d" % (4 << 30), "--cpu=20"]


class Inline(ast.NodeTransformer):
    def __init__(self, env):
        self.env = env

    def visit_Name(self, node):
        return copy.deepcopy(self.env[node.id]) if isinstance(node.ctx, ast.Load) and node.id in self.env else node


def split(code, names):
    """(argument list, [(term source, inputs it uses)]) or None when equation() is not assignments + one return."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "equation"), None)
    if fn is None:
        return None
    env, ret = {}, None
    for st in fn.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            env[st.targets[0].id] = Inline(env).visit(copy.deepcopy(st.value))
            continue
        if isinstance(st, ast.Return) and st.value is not None:
            ret = Inline(env).visit(copy.deepcopy(st.value))
            break
        return None
    if ret is None:
        return None
    terms = []

    def walk(node, sign):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            walk(node.left, sign)
            walk(node.right, sign if isinstance(node.op, ast.Add) else -sign)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            walk(node.operand, -sign)
        else:
            src = ast.unparse(ast.fix_missing_locations(node))
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & set(names)
            terms.append((("-(%s)" % src) if sign < 0 else "(%s)" % src, used))

    walk(ret, 1)
    return ast.unparse(fn.args), terms


def part(args, expr, free, train, tmp):
    f = Path(tmp) / "part.py"
    f.write_text("def equation(%s):\n    return %s\n" % (args, expr))
    try:
        p = subprocess.run(LIMIT + [PY, "-I", str(ISO / "fit_part.py"), "--code", str(f), "--train", train, "--free"] + free,
                           capture_output=True, text=True, timeout=45, env=ENV)
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, ValueError):
        return {"ok": False}


def show(srcs):
    s = " + ".join(srcs)
    return s if len(s) <= 160 else s[:157] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--train", required=True)
    a = ap.parse_args()
    names = open(a.train).readline().strip().split(",")[:-1]
    sp = split(open(a.code).read(), names)
    if sp is None:
        return {"ok": True, "text": None, "why": "equation() is not a single return of a sum"}
    args, terms = sp
    multi = [s for s, used in terms if len(used) >= 2]
    single = {v: [s for s, used in terms if used == {v}] for v in names}
    out = {"ok": True, "mixing": None, "single": {}, "n_terms": len(terms)}
    L = ["term check (each part is fitted together with free curves that stand in for the terms it leaves out):"]
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.code))) as tmp:
        r = part(args, " + ".join(multi) if multi else "0.0*" + names[0], names, a.train, tmp)
        if not r.get("ok") or r.get("share") is None:
            return {"ok": True, "text": None, "why": "the part check failed"}
        if multi:
            out["mixing"] = bool(r["right"])
            L.append("- the terms that mix inputs, %s: %s" % (show(multi), "RIGHT, keep this part as it is" if r["right"] else
                                                               "NOT right, change this part first"))
        else:
            out["mixing"] = False
            L.append("- your candidate has no term that mixes two inputs, but curves of single inputs alone leave %.1e "
                     "of the variance: a term that is a product of functions of two inputs is missing, find it first" % r["curves_alone"])
        if out["mixing"]:
            for v in names:
                r = part(args, " + ".join(multi + single[v]), [n for n in names if n != v], a.train, tmp)
                if not r.get("ok") or r.get("share") is None:
                    continue
                out["single"][v] = bool(r["right"])
                if single[v]:
                    L.append("- the terms in %s alone, %s: %s" % (v, show(single[v]), "RIGHT, keep this part as it is" if r["right"]
                                                                  else "NOT right, change this part"))
                else:
                    L.append("- no term in %s alone: %s" % (v, "RIGHT, none is needed" if r["right"] else
                                                            "NOT right, a term in %s alone is missing" % v))
        else:
            L.append("- the single-input terms cannot be checked before the mixing part is right")
    out["text"] = "\n".join(L)
    return out


if __name__ == "__main__":
    try:
        r = main()
    except Exception as e:
        r = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
    json.dump(r, sys.stdout)
