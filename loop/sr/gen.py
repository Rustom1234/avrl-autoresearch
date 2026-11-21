#!/usr/bin/env python3
"""Make equation-discovery problems that cannot be solved from memory.

    gen.py --out loop/sr/problems --seeds 0 1 2 3 --levels 1 2 3 4

One problem = one hidden law y = f(inputs) drawn fresh from a seed: a sum of 2 to 4 terms, some standard for the
field (powers, decay, saturation) and some unusual (nested or with a free exponent). The data has no noise, so the
right law fits to machine precision and any other law does not: the score is exact and does not depend on run time.
The level sets how hard the law is to find (number of terms, constants inside nonlinear functions, unusual factors).

Files per problem: card.json (what the solver is told), train.csv (what the solver may use), val.csv (same ranges,
used by the loop to decide that a target is reached), test_id.csv and test_ood.csv (held out; ood lies outside the
training ranges), truth.json (the law, never shown).

Same recipe as LSR-Synth of LLM-SRBench (Shojaee et al. 2025): known terms plus synthetic terms, checked for
well-posedness. Ours are generated here so that no instance can be in any model's training data.
"""
import argparse, json
from pathlib import Path

import numpy as np

DOMAINS = {
    "chem": {"context": "An isothermal reaction in a stirred tank. The rate was measured at many times and "
                        "concentrations by repeated experiments.",
             "target": ("dA_dt", "rate of change of the concentration of species A", "mol/(L s)"),
             "vars": [("t", "time since the start of the experiment", "s", (0.5, 10.0)),
                      ("A", "concentration of species A", "mol/L", (0.2, 4.0))]},
    "bio": {"context": "A population of a single species in a closed habitat. The growth rate was measured at many "
                       "times and population densities.",
            "target": ("dP_dt", "rate of change of the population density", "thousand per km^2 per day"),
            "vars": [("t", "time since the start of the observations", "days", (0.5, 12.0)),
                     ("P", "population density", "thousand per km^2", (0.3, 6.0))]},
    "phys": {"context": "A driven mass on a nonlinear spring with damping, observed on the outward stroke. The "
                        "acceleration was measured at many times, displacements and velocities.",
             "target": ("dv_dt", "acceleration of the mass", "m/s^2"),
             "vars": [("t", "time", "s", (0.5, 8.0)), ("x", "displacement", "m", (0.2, 3.0)),
                      ("v", "velocity", "m/s", (0.2, 3.0))]},
    "matsci": {"context": "Tensile tests of an alloy at several temperatures. The stress was recorded at many "
                          "strains and temperatures.",
               "target": ("sigma", "stress", "100 MPa"),
               "vars": [("eps", "strain", "percent", (0.2, 5.0)),
                        ("T", "temperature", "100 K", (2.0, 6.0))]},
}

# A factor is a function of one positive variable v on [a, b]. (name, python template, constants drawn given (a, b)).
# {v} is the variable, {c0} a constant placed inside the function.
SIMPLE = [("lin", "{v}", None), ("sq", "{v}**2", None), ("sqrt", "np.sqrt({v})", None), ("inv", "1/{v}", None),
          ("cube", "{v}**3", None), ("p15", "{v}**1.5", None)]
COMMON = [("expdecay", "np.exp(-{c0}*{v})", lambda r, a, b: [r.uniform(0.6, 3.0) / b]),
          ("satur", "{v}/(1 + {c0}*{v})", lambda r, a, b: [r.uniform(0.8, 4.0) / b]),
          ("hyper", "1/(1 + {c0}*{v})", lambda r, a, b: [r.uniform(0.8, 4.0) / b]),
          ("log1p", "np.log(1 + {c0}*{v})", lambda r, a, b: [r.uniform(1.0, 6.0) / b]),
          ("sin", "np.sin({c0}*{v})", lambda r, a, b: [r.uniform(2.5, 7.0) / b]),
          ("ricker", "{v}*np.exp(-{c0}*{v})", lambda r, a, b: [r.uniform(1.0, 4.0) / b])]
UNUSUAL = [("sinsqrt", "np.sin({c0}*np.sqrt({v}))", lambda r, a, b: [r.uniform(3.0, 7.0) / np.sqrt(b)]),
           ("expsqrt", "np.exp(-{c0}*np.sqrt({v}))", lambda r, a, b: [r.uniform(1.0, 3.0) / np.sqrt(b)]),
           ("freepow", "{v}**{c0}", lambda r, a, b: [r.choice([0.37, 0.62, 0.81, 1.23, 1.74, 2.31, 2.68])]),
           ("hill", "{v}**2/(1 + {c0}*{v}**2)", lambda r, a, b: [r.uniform(1.0, 6.0) / b ** 2]),
           ("gauss", "np.exp(-{c0}*{v}**2)", lambda r, a, b: [r.uniform(1.0, 4.0) / b ** 2]),
           ("tanh", "np.tanh({c0}*{v})", lambda r, a, b: [r.uniform(1.5, 5.0) / b]),
           ("vlogv", "{v}*np.log({v})", None), ("logsq", "np.log(1 + {v})**2", None)]
# level -> list of terms, a term being a tuple of factor pools (one pool = one factor, two = a product of two inputs)
LEVELS = {1: [(SIMPLE,), (SIMPLE,)],
          2: [(SIMPLE,), (SIMPLE, SIMPLE), (COMMON,)],
          3: [(SIMPLE,), (COMMON, SIMPLE), (UNUSUAL,)],
          4: [(SIMPLE,), (COMMON, SIMPLE), (UNUSUAL,), (UNUSUAL, SIMPLE)],
          # added after the first gate run (levels 3 and 4 were solved by no model): steps between 1 and 2, 2 and 3
          5: [(SIMPLE,), (COMMON,)],
          6: [(SIMPLE,), (SIMPLE, SIMPLE), (UNUSUAL,)]}
ORDER = [1, 5, 2, 6, 3, 4]      # levels from easiest to hardest
N = {"train": 300, "val": 100, "test_id": 200, "test_ood": 200}
OOD_EXT = 0.4      # ood: at least one input lies up to 40 percent of its range above the training range


def sig3(x):
    return float("%.3g" % x)


def draw_inputs(rng, vars_, n, ood):
    cols = {}
    hi = {name: b + OOD_EXT * (b - a) for name, _, _, (a, b) in vars_}
    for name, _, _, (a, b) in vars_:
        cols[name] = rng.uniform(a, hi[name] if ood else b, n)
    if ood:                 # force one input per row outside the training range
        pick = rng.integers(0, len(vars_), n)
        for j, (name, _, _, (a, b)) in enumerate(vars_):
            m = pick == j
            cols[name][m] = rng.uniform(b, hi[name], m.sum())
    return cols


def make(domain, level, seed):
    """Returns (card, truth, data) or None when this draw is not well posed (caller tries the next sub-seed)."""
    D = DOMAINS[domain]
    rng = np.random.default_rng([seed, level, sorted(DOMAINS).index(domain)])
    vars_ = D["vars"]
    rngs = {name: (a, b) for name, _, _, (a, b) in vars_}
    terms, used = [], set()
    for pools in LEVELS[level]:
        vs = list(rng.choice([v[0] for v in vars_], size=len(pools), replace=False))
        parts, key = [], []
        for pool, v in zip(pools, vs):
            name, tpl, draw = pool[rng.integers(len(pool))]
            c = [sig3(x) for x in draw(rng, *rngs[v])] if draw else []
            parts.append(tpl.format(v=v, c0=c[0] if c else ""))
            key.append((name, v))
        key = tuple(sorted(key))
        if key in used:
            return None
        used.add(key)
        terms.append("(" + ")*(".join(parts) + ")" if len(parts) > 1 else parts[0])
    data = {s: draw_inputs(rng, vars_, n, s == "test_ood") for s, n in N.items()}
    ev = lambda expr, cols: eval(expr, {"np": np}, cols)
    T = np.array([ev(t, data["train"]) for t in terms])
    if not np.all(np.isfinite(T)):
        return None
    share = rng.uniform(0.6, 1.4, len(terms))
    coefs = [sig3(rng.choice([-1, 1]) * s / T[i].std() * 2.0) for i, s in enumerate(share)]
    if any(abs(c) < 1e-3 or abs(c) > 1e3 for c in coefs):
        return None
    C = np.corrcoef(T) if len(terms) > 1 else np.eye(1)
    if np.max(np.abs(C - np.eye(len(terms)))) > 0.97:      # two terms nearly the same on the data: not identifiable
        return None
    expr = " + ".join("%r*%s" % (c, t) for c, t in zip(coefs, terms))
    y = {s: ev(expr, data[s]) for s in N}
    if not all(np.all(np.isfinite(v)) for v in y.values()) or y["train"].std() < 1e-6:
        return None
    contrib = np.array([abs(c) * T[i].std() for i, c in enumerate(coefs)]) / y["train"].std()
    if contrib.min() < 0.15:                                # every term must matter
        return None
    tname, tmean, tunit = D["target"]
    card = {"domain": domain, "context": D["context"],
            "target": {"name": tname, "meaning": tmean, "unit": tunit},
            "inputs": [{"name": n, "meaning": m, "unit": u, "train_range": [a, b]} for n, m, u, (a, b) in vars_],
            "n_train": N["train"]}
    truth = {"expr": expr, "level": level, "seed": seed, "n_terms": len(terms),
             "n_inner_constants": sum(t.count(".") > 0 and any(ch.isdigit() for ch in t) for t in terms)}
    return card, truth, {s: dict(data[s], **{tname: y[s]}) for s in N}


def write(out, pid, card, truth, data):
    d = Path(out) / pid
    d.mkdir(parents=True, exist_ok=True)
    cols = [i["name"] for i in card["inputs"]] + [card["target"]["name"]]
    for s, cd in data.items():
        M = np.column_stack([cd[c] for c in cols])
        np.savetxt(d / (s + ".csv"), M, delimiter=",", header=",".join(cols), comments="", fmt="%.12g")
    (d / "card.json").write_text(json.dumps(dict(card, id=pid), indent=1))
    (d / "truth.json").write_text(json.dumps(truth, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--domains", nargs="+", default=sorted(DOMAINS))
    a = ap.parse_args()
    for seed in a.seeds:
        for level in a.levels:
            for dom in a.domains:
                for sub in range(50):               # first well-posed draw of this (seed, level, domain)
                    r = make(dom, level, seed * 1000 + sub)
                    if r:
                        pid = "%s_L%d_s%d" % (dom, level, seed)
                        write(a.out, pid, *r)
                        print(pid, r[1]["expr"])
                        break
                else:
                    print("no well-posed draw for", dom, level, seed)
