#!/usr/bin/env python3
"""Analysis of run P2B (loop/design/P2B_PREREG.md): gates, ordered claims, verdict and the descriptive items.

    python -m loop.core.analysis_p2b --run loop/runs/P2B [--out DIR] [--pool-csv loop/runs/P2F/analysis/loops.csv]
    python -m loop.core.analysis_p2b --loops-csv loop/runs/P2F/analysis/loops.csv --out DIR    (check on stored P2 loops)

The per-loop table (one row per loop: cell, block, sealed gain, dollars) is built by analysis_p2.load_run; this file
only adds the P2B cells to it and applies the P2B rules. Reading order: endpoint (the metric), bootstrap (its
intervals), signflip (the tests), decide (gates, claims, verdict), build_report.

Terms:
  anchors        SS (all strong) and QT (all cheap). Every policy is scored between them.
  gap recovered  (gain(P) - gain(QT)) / (gain(SS) - gain(QT)): the share of the strong loop's extra gain that P keeps.
  extra cost     the same ratio on model dollars per loop: the share of the strong loop's extra spend that P pays.
  advantage      gap recovered - extra cost. 0 is the straight line between the anchors (what mixing the two anchors
                 at random gives), 1 is the strong loop's gain at the cheap loop's price.
  block          one loop of every cell, run close together in time. Tests and the bootstrap work on blocks.
"""
import argparse
import itertools
import json
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis as A  # noqa: E402
import analysis_p2 as P2A  # noqa: E402
import plan_p2b  # noqa: E402

# Constants of the decision rules, as written in P2B_PREREG (Outcome and endpoint, Decision rules).
#   alpha        one-sided level of every sign-flip test
#   noise_sd     one noise sd of val_bpb; a bootstrap draw whose anchor gap is at or below it counts as "no gap"
#   small_max    G2 needs fewer than this share of draws with no gap
#   ni_margin    C3: the 90 percent interval of gain(ST) - gain(SS) must lie above this
#   n_boot, ci   block bootstrap draws and interval level; boot_seed fixes the draws
K = dict(alpha=0.05, noise_sd=0.004, small_max=0.05, ni_margin=-0.010, n_boot=10000, ci=0.90, boot_seed=1)
GAIN, COST, COST_GPU = "sealed_gain_global", "model_usd", "total_usd_gpu_2"
ANCHORS = ("SS", "QT")
POLICIES = ("ST", "F", "M", "R", "H")
# planner and coder labels of the cells P2 did not have (only used as columns of loops.csv)
NEW_FACTORS = {"F": ("F", "T"), "R": ("R", "R"), "M": ("M", "T"), "N": ("N", "N")}
# the seven coins of an R table that the engine reads (planner and planner_cheap are unused fallbacks)
LIVE_KEYS = tuple(k for k in plan_p2b.R_KEYS if k not in ("planner", "planner_cheap"))


@contextmanager
def p2b_cells():
    """Lets analysis_p2.load_run accept the F, R and M loops while the block runs."""
    cells, factors = P2A.CELLS, P2A.FACTORS
    P2A.CELLS, P2A.FACTORS = cells + tuple(NEW_FACTORS), {**factors, **NEW_FACTORS}
    try:
        yield
    finally:
        P2A.CELLS, P2A.FACTORS = cells, factors


def block_table(df, col):
    """Blocks x cells table of col over the finished loops, keeping only blocks that have every cell of the run."""
    t = df[df.status == "done"].pivot_table(index="block", columns="cell", values=col, aggfunc="mean")
    return t.dropna()


def endpoint(g, c, policy):
    """(gap recovered, extra cost, advantage) of a policy from the cell means of gain (g) and cost (c)."""
    rec = (g[policy] - g["QT"]) / (g["SS"] - g["QT"])
    ext = (c[policy] - c["QT"]) / (c["SS"] - c["QT"])
    return rec, ext, rec - ext


def signflip(d, alternative):
    """One-sided sign-flip permutation p-value for the mean of the within-block differences d. Under the null each
    block's difference is as likely to be +d as -d. Exact (all 2**n sign patterns) up to 16 blocks, 200,000 random
    patterns beyond that. alternative is 'greater' (mean above 0) or 'less'."""
    d = np.asarray(d, float)
    n = len(d)
    if n <= 16:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        means = signs @ d / n
        extra = 0
    else:
        signs = np.random.default_rng(K["boot_seed"]).choice((1.0, -1.0), size=(200000, n))
        means = signs @ d / n
        extra = 1
    obs = d.mean()
    hits = (means >= obs - 1e-12).sum() if alternative == "greater" else (means <= obs + 1e-12).sum()
    return float((hits + extra) / (len(means) + extra))


def bootstrap(gt, ct, policies):
    """Block bootstrap of the endpoint. gt and ct are the blocks x cells tables of gain and cost. Returns
    {policy: {recovered, extra_cost, advantage: [point, low, high]}}, the share of draws whose anchor gap is at or
    below one noise sd, and the draws of the cell mean gains (for the gain-difference intervals)."""
    rng = np.random.default_rng(K["boot_seed"])
    idx = rng.integers(0, len(gt), size=(K["n_boot"], len(gt)))
    gm = pd.DataFrame(gt.values[idx].mean(axis=1), columns=gt.columns)
    cm = pd.DataFrame(ct.values[idx].mean(axis=1), columns=ct.columns)
    lo, hi = 50 * (1 - K["ci"]), 100 - 50 * (1 - K["ci"])
    out = {}
    for p in policies:
        point, draws = endpoint(gt.mean(), ct.mean(), p), endpoint(gm, cm, p)
        out[p] = {name: [float(point[i])] + [float(v) for v in np.percentile(draws[i], [lo, hi])]
                  for i, name in enumerate(("recovered", "extra_cost", "advantage"))}
    return out, float(((gm["SS"] - gm["QT"]) <= K["noise_sd"]).mean()), gm


def gain_diff(gt, gm, a, b):
    """[point, low, high] of gain(a) - gain(b): cell means and the bootstrap interval."""
    lo, hi = 50 * (1 - K["ci"]), 100 - 50 * (1 - K["ci"])
    return [float(gt[a].mean() - gt[b].mean())] + [float(v) for v in np.percentile(gm[a] - gm[b], [lo, hi])]


def decide(gt, boot, small, gm):
    """Gates, the ordered claims and the verdict. A claim is 'held', 'not held', or 'not tested' when an earlier claim
    (or a gate) failed. C4 is reported beside the verdict and does not change it."""
    cells = set(gt.columns)
    gates = {
        "G1": dict(what="damaged loop detected: gain(H) below gain(SS)", p=signflip(gt["H"] - gt["SS"], "less")),
        "G2": dict(what="anchors differ: gain(SS) above gain(QT)", p=signflip(gt["SS"] - gt["QT"], "greater"),
                   share_small_gap=small),
    }
    gates["G1"]["held"] = gates["G1"]["p"] < K["alpha"]
    gates["G2"]["held"] = gates["G2"]["p"] < K["alpha"] and small < K["small_max"]
    c3 = gain_diff(gt, gm, "ST", "SS")
    tests = [
        ("C1", "fixed table beats the line: interval of ST's advantage above 0", boot["ST"]["advantage"][1] > 0,
         dict(advantage=boot["ST"]["advantage"])),
        ("C2", "planner effect: gain(ST) above gain(QT)", None, dict(p=signflip(gt["ST"] - gt["QT"], "greater"))),
        ("C3", "cheap coder non-inferior: interval of gain(ST) - gain(SS) above %.3f" % K["ni_margin"],
         c3[1] > K["ni_margin"], dict(diff=c3)),
    ]
    if "F" in cells:
        tests.append(("C4", "first-list rule beats the line: interval of F's advantage above 0",
                      boot["F"]["advantage"][1] > 0, dict(advantage=boot["F"]["advantage"])))
    claims, open_ = {}, all(g["held"] for g in gates.values())
    for name, what, ok, info in tests:
        ok = info["p"] < K["alpha"] if ok is None else ok
        claims[name] = dict(what=what, status=("held" if ok else "not held") if open_ else "not tested", **info)
        open_ = open_ and bool(ok)
    st = {k: v["status"] for k, v in claims.items()}
    if not all(g["held"] for g in gates.values()):
        verdict = "KILL"
    elif st["C1"] != "held":
        verdict = "NOT REPLICATED"
    elif st["C2"] == st["C3"] == "held":
        verdict = "REPLICATED"
    else:
        verdict = "PARTIAL (%s not held)" % ("C2" if st["C2"] != "held" else "C3")
    return dict(gates=gates, claims=claims, verdict=verdict)


def r_coins(df, seed):
    """Descriptive: gain of the R loops against their coin flips. For each of the seven live keys, the mean gain of the
    loops whose coin gave the strong model minus the mean of the others, and the least-squares coefficient of that
    coin with all seven in one model (12 loops, 8 parameters, so the coefficients are rough)."""
    r = df[(df.cell == "R") & (df.status == "done") & df[GAIN].notna()]
    if len(r) < 9:
        return None
    X = np.array([[plan_p2b.r_draws(seed, int(b))[k] == "SS" for k in LIVE_KEYS] for b in r.block], float)
    y = r[GAIN].to_numpy(float)
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0][1:]
    out = {}
    for j, k in enumerate(LIVE_KEYS):
        s, q = y[X[:, j] == 1], y[X[:, j] == 0]
        out[k] = dict(n_strong=len(s), mean_diff=float(s.mean() - q.mean()) if len(s) and len(q) else None,
                      ols=float(coef[j]))
    return out


def analyze(df, seed=None, pool=None):
    """Everything from the per-loop table: results dict. pool is P2's loops table for the pooled estimate."""
    # Cell N (no-LLM random search) runs in its own later session and is descriptive: it is kept out of the tables
    # behind the verdict, so a missing N loop can never drop a block from the confirmatory analysis.
    full, df = df, df[df.cell != "N"]
    gt, ct, ctg = block_table(df, GAIN), block_table(df, COST), block_table(df, COST_GPU)
    pols = [p for p in POLICIES if p in gt.columns]
    boot, small, gm = bootstrap(gt, ct, pols)
    res = dict(n_blocks=len(gt), blocks_dropped=int(df.block.nunique() - len(gt)), constants=K,
               cell_means={c: dict(gain=float(gt[c].mean()), model_usd=float(ct[c].mean()),
                                   total_usd_gpu_2=float(ctg[c].mean())) for c in gt.columns},
               endpoint=boot, endpoint_total_dollars=bootstrap(gt, ctg, pols)[0], share_small_gap=small,
               **decide(gt, boot, small, gm))
    desc = [c for c in ("stall", "t_2e17", "usd_per_backed_change", "served_fraction_ledger", "planner_served_fraction",
                        "n_fallbacks", "harm_noops", "n_jobs", "gpu_hours") if c in df]
    res["by_cell"] = df[df.status == "done"].groupby("cell")[desc].mean().round(4).to_dict("index")
    if seed is not None and "R" in gt.columns:
        res["r_coins"] = r_coins(df, seed)
    if (full.cell == "N").any():
        gn, cn, cgn = block_table(full, GAIN), block_table(full, COST), block_table(full, COST_GPU)
        res["cell_n"] = dict(n_blocks=len(gn), gain=float(gn["N"].mean()), endpoint=bootstrap(gn, cn, ["N"])[0]["N"],
                             endpoint_total_dollars=bootstrap(gn, cgn, ["N"])[0]["N"],
                             p_ss_above_n=signflip((gn["SS"] - gn["N"]).to_numpy(), "greater"),
                             p_n_above_qt=signflip((gn["N"] - gn["QT"]).to_numpy(), "greater"))
    if pool is not None:
        both = [c for c in ("SS", "QT", "ST", "H") if c in gt.columns]
        p2 = pool[pool.cell.isin(both)].assign(block=lambda d: d.block + 1000)
        cat = pd.concat([p2, df[df.cell.isin(both)]])
        res["p2_alone"] = bootstrap(block_table(p2, GAIN), block_table(p2, COST), ["ST", "H"])[0]
        res["pooled_p2_p2b"] = bootstrap(block_table(cat, GAIN), block_table(cat, COST), ["ST", "H"])[0]
    return res


def f3(v):
    return "%.2f (%.2f to %.2f)" % tuple(v)


def build_report(res):
    """report.md: verdict first, then gates, claims, the endpoint table and the descriptive items."""
    L = ["# P2B analysis", "", "**Verdict: %s**" % res["verdict"], "",
         "%d complete blocks (%d dropped for a missing loop). Outcome: sealed held-out gain. Cost: model dollars per loop."
         % (res["n_blocks"], res["blocks_dropped"]), "", "## Gates", ""]
    for k, g in res["gates"].items():
        extra = ", share of draws with no anchor gap %.3f" % g["share_small_gap"] if "share_small_gap" in g else ""
        L.append("- **%s, %s:** p = %.4f%s. %s." % (k, g["what"], g["p"], extra, "Held" if g["held"] else "FAILED"))
    L += ["", "## Claims (in order)", ""]
    for k, c in res["claims"].items():
        num = "p = %.4f" % c["p"] if "p" in c else "advantage %s" % f3(c["advantage"]) if "advantage" in c \
            else "difference %.4f (%.4f to %.4f)" % tuple(c["diff"])
        L.append("- **%s, %s:** %s. %s." % (k, c["what"], num, c["status"].capitalize()))
    L += ["", "## Endpoint by policy (point and 90 percent block-bootstrap interval)", "",
          "| Policy | Gap recovered | Extra cost | Advantage | Advantage, GPU dollars added |", "|---|---|---|---|---|"]
    for p, e in res["endpoint"].items():
        L.append("| %s | %s | %s | %s | %s |" % (p, f3(e["recovered"]), f3(e["extra_cost"]), f3(e["advantage"]),
                                                f3(res["endpoint_total_dollars"][p]["advantage"])))
    L += ["", "Reading: 0 is the straight line between all cheap and all strong, 1 is the strong gain at the cheap price.",
          "", "## Cell means", "", "| Cell | Sealed gain | Model USD per loop | Model + GPU USD |", "|---|---|---|---|"]
    for c, m in res["cell_means"].items():
        L.append("| %s | %.4f | %.2f | %.2f |" % (c, m["gain"], m["model_usd"], m["total_usd_gpu_2"]))
    if res.get("cell_n"):
        n = res["cell_n"]
        L += ["", "## Cell N: random search with no language model (descriptive)", "",
              "Run in its own session after the blocks, scored against that session's own confirmation anchors; "
              "%d blocks." % n["n_blocks"], "",
              "- Sealed gain %.4f. Gap recovered %s, extra cost %s, advantage %s."
              % (n["gain"], f3(n["endpoint"]["recovered"]), f3(n["endpoint"]["extra_cost"]), f3(n["endpoint"]["advantage"])),
              "- Advantage with GPU dollars added: %s." % f3(n["endpoint_total_dollars"]["advantage"]),
              "- gain(SS) above gain(N): one-sided p = %.4f. gain(N) above gain(QT): one-sided p = %.4f."
              % (n["p_ss_above_n"], n["p_n_above_qt"]),
              "- If N recovers most of the gap, the task is too simple to rank loop designs."]
    if res.get("r_coins"):
        L += ["", "## Random tables: gain against each coin (descriptive)", "",
              "| Key | Loops with the strong model | Mean gain, strong minus cheap | Least-squares coefficient |", "|---|---|---|---|"]
        for k, v in res["r_coins"].items():
            L.append("| %s | %d | %s | %.4f |" % (k, v["n_strong"], "n/a" if v["mean_diff"] is None else "%.4f" % v["mean_diff"], v["ols"]))
    if "pooled_p2_p2b" in res:
        L += ["", "## P2 and P2B side by side (advantage)", "", "| Policy | P2 | P2B | Pooled |", "|---|---|---|---|"]
        for p in ("ST", "H"):
            L.append("| %s | %s | %s | %s |" % (p, f3(res["p2_alone"][p]["advantage"]), f3(res["endpoint"][p]["advantage"]),
                                               f3(res["pooled_p2_p2b"][p]["advantage"])))
    L += ["", "## Other measures by cell (means, descriptive)", "", "```", pd.DataFrame(res["by_cell"]).T.to_string(), "```", ""]
    return "\n".join(L).replace(P2A.EM_DASH, "-")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=None, help="run directory with plan.json and the loop directories")
    ap.add_argument("--loops-csv", default=None, help="use a stored per-loop table instead of reading a run directory")
    ap.add_argument("--pool-csv", default=None, help="P2's loops.csv, for the side-by-side and pooled estimates")
    ap.add_argument("--n-run", default=None, help="run directory of cell N (random search), added as descriptive")
    ap.add_argument("--prices", default=str(P2A.DEFAULT_PRICES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    seed = None
    if a.loops_csv:
        df = pd.read_csv(a.loops_csv)
    else:
        with p2b_cells():
            df = P2A.load_run(a.run, json.loads(Path(a.prices).read_text()))[0]
        seed = (A.read_json(Path(a.run) / "plan.json") or {}).get("seed")
    if a.n_run:
        with p2b_cells():
            df = pd.concat([df, P2A.load_run(a.n_run, json.loads(Path(a.prices).read_text()))[0]], ignore_index=True)
    res = analyze(df, seed, pd.read_csv(a.pool_csv) if a.pool_csv else None)
    out = Path(a.out or Path(a.run) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "loops.csv", index=False)
    (out / "results.json").write_text(json.dumps(A.clean_json(res), indent=1))
    (out / "report.md").write_text(build_report(res))
    print("verdict %s; %d blocks; wrote %s" % (res["verdict"], res["n_blocks"], out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
