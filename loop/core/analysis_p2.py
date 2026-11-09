#!/usr/bin/env python3
"""Analysis of the P2 run (loop/design/P2_PREREG.md): planner x coder factorial plus the harm control.

    python -m loop.core.analysis_p2 --run DIR [--out DIR/analysis] [--prices loop/models/open_prices.json]
                                    [--cell-map C=SS,D=ST]

Reads, per loop dir: state.json, ledger.jsonl, calls.jsonl, config.json, incumbent.py, slots/<k>/result.json and the
slot sources of accepted changes; and per run: plan.json (loop cells), anchor.json, confirmation/results.jsonl,
sentinels.jsonl, launch_sentinels.jsonl. Writes report.md, loops.csv and results.json. --cell-map labels loops of an
older run by loop id prefix (C=SS,D=ST) and drops the rest; it exists to smoke-test the code on R1, whose results
mean nothing for P2.

Where it sits: the last step of P2. plan_p2.py lays out the loops, anchor_p2.py freezes the baseline, the controller
runs the loops, confirmation.py re-runs the final incumbents, and this script applies the pre-registered decision
rules. Reading order: load_loop (one table row per loop), effect and harm_control (the tests), decide (the verdict),
build_report. The permutation test, confirmation pairing and served-fraction helpers are reused from analysis.py.

Terms:
  val_bpb             validation bits per byte after the 300 second training run; lower is better.
  slot                one GPU training run charged to a loop's budget (20 per loop).
  loop                one automated research loop: an LLM planner proposes changes to train.py, an LLM coder writes
                      the edit, and an acceptance rule decides whether it replaces the incumbent (the current best).
  cell                one routing configuration, named by two letters. First letter = planner (S Sonnet, Q the small
                      open model qwen35-4b), second = coder (S Sonnet, T the cheap per-class table). SS is all strong, QT all
                      cheap. H is the harm control: SS with half of the coder's edits replaced by a no-op. It is a
                      positive control: if the measures cannot tell H from SS, they cannot rank the other cells.
  main effect         planner: Q loops minus S loops (QS and QT against SS and ST); coder: T minus S. H is left out.
  2**17               in the earlier run R1 the loops that never lowered TOTAL_BATCH_SIZE to 2**17 or below were the
                      ones with the smaller held-out gain, so reaching it is the progress marker of this substrate.
  sealed gain         gain over the unmodified baseline (the anchor) on a held-out shard the loop never saw.

Definitions (one line each, all from P2_PREREG and the design decisions frozen before the run):
  stall               final incumbent TOTAL_BATCH_SIZE > 2**17 (AST rule of r1_data.md, last module-level assignment).
  time to 2**17       slots used (0-based slot index + 1) up to and including the first accepted change whose source has
                      TOTAL_BATCH_SIZE <= 2**17; a loop that never gets there is censored at its number of finished
                      slots (20 for a finished loop). The slot margins (5 slots) use the Kaplan-Meier MEDIAN of this
                      time per group, as P2_PREREG says; the mean is reported next to it. A median that is not reached
                      (more than half of a group censored) counts as +inf, so it can trigger PIVOT but never PASS.
  time to last change slots used up to and including the last accepted change; a loop with no accepted change is
                      censored at its slots.
  backed change       an accepted change that rests on at least two runs with different seeds (incumbent_after.n >= 2).
  load-flagged job    a slot job whose result.json has load_flagged (another process was on its GPU); a sensitivity
                      verdict is reported on the loops that have no such job.
"""
import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import analysis as A  # noqa: E402  (permutation test with intervals, confirm_summary, served-fraction helpers)
import cost_report  # noqa: E402

# Margins and constants of the decision rules, as written in P2_PREREG (Analysis and Decision rules).
#   harm_alpha      one-sided level of the harm control test (H slower than SS)
#   stall_margin    a cheap role "holds" when it raises the stall rate by less than this (rate difference)
#   slots_margin    ... and delays the median time to 2**17 by less than this many slots
#   cost_ratio_min  PASS also needs QT at least this many times cheaper than SS per backed change
#   stall_cut       2**17, the TOTAL_BATCH_SIZE cut of the stall rule
#   censor_slot     slots per loop: the censoring time of a loop that never reaches an event
#   n_perm          random permutations per test when the exact enumeration is too large
#   ci_level        level of the permutation intervals
#   gpu_prices      USD per GPU-hour grid for the cost columns that add GPU time
CONSTANTS = json.loads("""
{"harm_alpha": 0.05, "stall_margin": 0.25, "slots_margin": 5.0, "cost_ratio_min": 5.0,
 "stall_cut": 131072, "censor_slot": 20, "n_perm": 10000, "ci_level": 0.90, "gpu_prices": [0.5, 1, 2, 4]}
""")
CELLS = ("SS", "ST", "QS", "QT", "H")
# (planner level, coder level) of each cell; H routes like SS and is kept apart by its harm flag
FACTORS = {"SS": ("S", "S"), "ST": ("S", "T"), "QS": ("Q", "S"), "QT": ("Q", "T"), "H": ("S", "S")}
DEFAULT_PRICES = REPO / "loop" / "models" / "open_prices.json"
EM_DASH = chr(0x2014)   # replaced by "-" in every output file, whatever the inputs contained


# ------------------------------------------------------------------ the big changes in a train.py
# The outcome measures read a handful of module-level constants from a loop's train.py. The file is parsed, never
# imported or executed, so a broken or hostile candidate cannot run code here.
def _const(node):
    """Value of a constant arithmetic expression (Constant, unary minus, + - * // ** <<), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.BinOp):
        a, b = _const(node.left), _const(node.right)
        if a is None or b is None:
            return None
        try:
            if isinstance(node.op, ast.Pow) and abs(b) > 64:   # refuse huge exponents instead of computing them
                return None
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.FloorDiv: lambda: a // b, ast.Pow: lambda: a ** b, ast.LShift: lambda: a << b}
            return ops[type(node.op)]()
        except (KeyError, ZeroDivisionError, TypeError, ValueError):
            return None
    return None


def parse_big_changes(src):
    """{TOTAL_BATCH_SIZE, DEPTH, ASPECT_RATIO, HEAD_DIM, width}: the value of the last module-level assignment of each
    name (None when absent or not a constant expression); width as the train.py computes it. All None when src is None
    or does not parse."""
    out = {k: None for k in ("TOTAL_BATCH_SIZE", "DEPTH", "ASPECT_RATIO", "HEAD_DIM")}
    try:
        tree = ast.parse(src or "")
    except (SyntaxError, ValueError):
        return dict(out, width=None)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in out:
            out[node.targets[0].id] = _const(node.value)
    d, ar, hd = out["DEPTH"], out["ASPECT_RATIO"], out["HEAD_DIM"]
    width = None
    if all(isinstance(v, int) for v in (d, ar, hd)) and hd > 0:
        width = ((d * ar + hd - 1) // hd) * hd   # DEPTH x ASPECT_RATIO rounded up to a multiple of HEAD_DIM
    return dict(out, width=width)


# ------------------------------------------------------------------ statistics
def logrank(ta, ea, tb, eb):
    """Two-sample log-rank test with right censoring (event flags ea, eb). z > 0 means group a had more events than
    expected, i.e. reached the event sooner. p_two is two-sided; p_a_slower = P(Z <= z) is the one-sided p for
    'a is slower than b'."""
    ta, tb = np.asarray(ta, float), np.asarray(tb, float)
    ea, eb = np.asarray(ea, bool), np.asarray(eb, bool)
    o_minus_e = var = 0.0
    for t in np.unique(np.concatenate([ta[ea], tb[eb]])):   # every distinct event time
        na, nb = (ta >= t).sum(), (tb >= t).sum()   # still at risk just before t
        da, db = ((ta == t) & ea).sum(), ((tb == t) & eb).sum()   # events at t
        n, d = na + nb, da + db
        o_minus_e += da - d * na / n   # observed minus expected events in a if both groups shared one hazard
        if n > 1:
            var += d * (na / n) * (1 - na / n) * (n - d) / (n - 1)   # hypergeometric variance of da
    if not var > 0:
        return dict(z=math.nan, o_minus_e=o_minus_e, var=var, p_two=math.nan, p_a_slower=math.nan)
    z = o_minus_e / math.sqrt(var)
    return dict(z=z, o_minus_e=o_minus_e, var=var, p_two=math.erfc(abs(z) / math.sqrt(2)),
                p_a_slower=0.5 * math.erfc(-z / math.sqrt(2)))


def km_median(t, e):
    """Kaplan-Meier median of right-censored times (event flags e): the smallest event time at which the survival curve is
    at or below 0.5, math.inf when it never gets there (more than half of the sample is censored first), NaN when empty."""
    t, e = np.asarray(t, float), np.asarray(e, bool)
    if not len(t):
        return math.nan
    surv = 1.0
    for u in np.unique(t[e]):
        surv *= 1 - ((t == u) & e).sum() / (t >= u).sum()
        if surv <= 0.5 + 1e-12:
            return float(u)
    return math.inf


def perm(x, y, seed, interval=False):
    """Two-sided permutation test of mean(y) - mean(x) with CONSTANTS['n_perm'] random draws (exact when the number of
    splits is small), plus the test-inversion interval of analysis.perm_test. analysis.N_MC is set for the call."""
    old, A.N_MC = A.N_MC, int(CONSTANTS["n_perm"])
    try:
        return A.perm_test(x, y, np.random.default_rng(seed), interval=interval, alpha=1 - CONSTANTS["ci_level"])
    finally:
        A.N_MC = old


def fisher(k_hi, n_hi, k_lo, n_lo, alternative="two-sided"):
    """Fisher exact p for two rates, k_hi of n_hi against k_lo of n_lo (NaN when a group is empty). Exact because
    the groups are small (12 loops per cell), too few for a chi-square approximation."""
    if n_hi < 1 or n_lo < 1:
        return math.nan
    return float(stats.fisher_exact([[k_hi, n_hi - k_hi], [k_lo, n_lo - k_lo]], alternative=alternative)[1])


# ------------------------------------------------------------------ per loop
def accepts_of(recs):
    """The ledger records whose decision is an accept, i.e. the changes that became the incumbent."""
    return [r for r in recs if str(r.get("decision") or "").lower().startswith("accept")]


def loop_cell(lid, plan_cells, config, cell_map):
    """The cell of loop lid, or None when it is not part of the experiment. With --cell-map only the map counts
    (prefix of the id before '_'); otherwise plan.json, then the loop's config.json, then an id such as SS_01."""
    if cell_map:
        return cell_map.get(lid.split("_")[0])
    cell = plan_cells.get(lid) or config.get("cell")
    if cell is None:
        m = re.match(r"^(%s)_" % "|".join(CELLS), lid)
        cell = m.group(1) if m else None
    return cell if cell in CELLS else None


def sealed_gains(conf):
    """({loop_id: global-anchor gain}, {loop_id: same-wave paired gain}, anchor sealed mean). Global version: mean sealed
    of all usable anchor runs minus the loop's mean sealed over its usable confirmation runs."""
    empty = ({}, {}, math.nan)
    if conf.empty:
        return empty
    c = conf.reindex(columns=["wave", "loop_id", "kind", "official_val_bpb", "sealed_val_bpb", "status"]).copy()
    c["sealed_val_bpb"] = pd.to_numeric(c.sealed_val_bpb, errors="coerce")
    c = c[(c.status.astype(str) == "ok") & c.sealed_val_bpb.notna()]
    anchor, runs = c[c.kind == "anchor"], c[c.kind == "confirm"]
    if anchor.empty or runs.empty:
        return empty
    glob = float(anchor.sealed_val_bpb.mean())
    g = {lid: glob - float(v) for lid, v in runs.groupby("loop_id").sealed_val_bpb.mean().items()}
    paired = {lid: float(v) for lid, v in A.confirm_summary(conf).improvement.items()}
    return g, paired, glob


def planner_served(recs, table):
    """(planner calls, calls served by the model the table names for their role) over the ledger model_calls whose role
    starts with 'planner' (backlog, widen, refresh). The ledger-based served fraction of analysis.py leaves these out, but
    they are the main calls of the planner factor. A planner role the table does not name goes to sonnet, as in the engine."""
    n = ok = 0
    for r in recs:
        for c in r.get("model_calls") or []:
            role = str(c.get("role") or "").lower()
            if role.startswith("planner"):
                want = str(A.designated_model(table, role, None) or "sonnet").lower()
                got = str(c.get("served_model") or "").lower()
                n += 1
                ok += bool(got) and want in got
    return n, ok


def load_loop(d, cell, prices, conf_gain, plan_loop):
    """One loop directory as (row, gpu_rows). row is the loop's line of loops.csv: the outcome measures (stall, the
    two event times with their event flags, accepts and backed accepts), dollars, served fractions, harm no-ops and
    sealed gains. gpu_rows has one dict per slot job for the per-GPU table.

    d is the loop directory, prices the open-model price file, conf_gain the two {loop_id: gain} dicts of
    sealed_gains, plan_loop the loop's plan.json entry (or None)."""
    st = A.read_json(d / "state.json") or {}
    recs = A.read_jsonl(d / "ledger.jsonl")
    config = A.read_json(d / "config.json") or {}
    lid = st.get("loop_id") or d.name
    planner, coder = FACTORS[cell]
    n_slots = int(A.num0(st.get("n_slots")) or CONSTANTS["censor_slot"])
    slots = A.num(st.get("slot_index"))   # slots finished so far; under n_slots for a loop that stopped early
    slots = n_slots if math.isnan(slots) else int(slots)
    row = dict(loop_id=lid, cell=cell, planner=planner, coder=coder, harm=cell == "H",
               block=(plan_loop or {}).get("block", config.get("block")), status=st.get("status"), slots_completed=slots)

    # stall: read from the final incumbent, so a loop that stopped early is judged on the incumbent it had
    inc = d / "incumbent.py"
    fin = parse_big_changes(inc.read_text() if inc.is_file() else None)
    tbs, depth = fin["TOTAL_BATCH_SIZE"], fin["DEPTH"]
    row.update(final_tbs=tbs, final_depth=depth, final_width=fin["width"],
               stall=math.nan if tbs is None else float(tbs > CONSTANTS["stall_cut"]))

    # first slot whose accepted source has TOTAL_BATCH_SIZE <= 2**17 (read from the slot's own train.py)
    acc = accepts_of(recs)
    first17 = None
    for r in acc:
        src = d / "slots" / str(r.get("slot")) / "train.py"
        big = parse_big_changes(src.read_text()) if src.is_file() else {}
        if big.get("TOTAL_BATCH_SIZE") is not None and big["TOTAL_BATCH_SIZE"] <= CONSTANTS["stall_cut"]:
            first17 = r["slot"] if first17 is None else min(first17, r["slot"])
    last = max((r["slot"] for r in acc), default=None)
    cap = min(slots, int(CONSTANTS["censor_slot"]))
    # event times are slots used (index + 1), so that they are on the same scale as the censoring time (finished slots)
    row.update(last_change_slot=last, first_2e17_slot=first17,
               event_last_change=last is not None, t_last_change=last + 1 if last is not None else cap,
               event_2e17=first17 is not None, t_2e17=first17 + 1 if first17 is not None else cap,
               accepts=len(acc), accepts_backed=sum(1 for r in acc if (r.get("incumbent_after") or {}).get("n", 0) >= 2))

    # dollars: every model call of the loop (Claude at list price, open models per token), GPU time from the slot jobs
    calls = A.read_jsonl(d / "calls.jsonl")
    c = cost_report.price_calls(calls, prices)
    jobs = [A.read_json(p) or {} for p in sorted((d / "slots").glob("*/result.json"))] if (d / "slots").is_dir() else []
    wall = sum(A.num0(j.get("wall_seconds")) for j in jobs)
    gpu_h = wall / 3600
    row.update(claude_usd=c["claude_usd"], open_usd=c["open_usd"], model_usd=c["claude_usd"] + c["open_usd"],
               n_jobs=len(jobs), gpu_hours=gpu_h, n_load_flagged=sum(1 for j in jobs if j.get("load_flagged")))
    for p in CONSTANTS["gpu_prices"]:
        row["total_usd_gpu_%s" % str(p).replace(".", "p")] = row["model_usd"] + gpu_h * p
    row["usd_per_backed_change"] = row["model_usd"] / row["accepts_backed"] if row["accepts_backed"] else math.nan

    # manipulation checks: did the models named in the cell's routing table actually serve the calls?
    table = (A.read_routing(config.get("routing")) or (None, None))[0]
    des, srv = A.num0(st.get("designated_open_calls")), A.num0(st.get("open_served_calls"))
    ldes, lsrv = A.ledger_served(recs, table) if table else (0, 0)
    pcalls, psrv = planner_served(recs, table) if table else (0, 0)
    row.update(served_fraction_state=srv / des if des > 0 else math.nan,
               served_fraction_ledger=lsrv / ldes if ldes > 0 else math.nan, n_fallbacks=len(st.get("fallbacks") or []),
               planner_calls=pcalls, planner_served_fraction=psrv / pcalls if pcalls > 0 else math.nan)
    noops = st.get("harm_noops")
    row["harm_noops"] = len(noops) if noops is not None else len(list((d / "slots").glob("*/harm.json")))
    g, p = conf_gain
    row.update(sealed_gain_global=g.get(lid, math.nan), sealed_gain_paired=p.get(lid, math.nan))
    gpu_rows = [dict(loop_id=lid, cell=cell, gpu=j.get("gpu"), wall_s=A.num(j.get("wall_seconds")),
                     flagged=bool(j.get("load_flagged"))) for j in jobs]
    return row, gpu_rows


COST_NAMES = {"model_usd": "model dollars (Claude + open)"}
COST_NAMES.update({"total_usd_gpu_%s" % str(p).replace(".", "p"): "model + GPU at %s USD/h" % p for p in CONSTANTS["gpu_prices"]})


def load_run(run, prices, cell_map=None):
    """Every loop of a run directory: (loops DataFrame, per-job GPU DataFrame, anchor sealed mean, skipped ids).
    A subdirectory counts as a loop when it has a state.json; one without a cell is listed in skipped."""
    run = Path(run)
    plan = A.read_json(run / "plan.json") or {}
    plan_loops = {l["loop_id"]: l for l in plan.get("loops", [])}
    plan_cells = {k: v.get("cell") for k, v in plan_loops.items()}
    conf = pd.DataFrame(A.read_jsonl(run / "confirmation" / "results.jsonl"))
    gains = sealed_gains(conf)
    rows, gpu_rows, skipped = [], [], []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and (p / "state.json").exists()):
        cfg = A.read_json(d / "config.json") or {}
        cell = loop_cell(d.name, plan_cells, cfg, cell_map)
        if cell is None:
            skipped.append(d.name)
            continue
        row, gr = load_loop(d, cell, prices, gains[:2], plan_loops.get(d.name))
        rows.append(row)
        gpu_rows += gr
    df = pd.DataFrame(rows)
    for col in ("last_change_slot", "first_2e17_slot"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df, pd.DataFrame(gpu_rows, columns=["loop_id", "cell", "gpu", "wall_s", "flagged"]), gains[2], skipped


# ------------------------------------------------------------------ tests and decisions
def group(df, factor, level):
    """Non-harm loops with the given planner or coder level."""
    return df[(~df.harm) & (df[factor] == level)]


def survival(hi, lo, col):
    """Time-to-event comparison of two groups of loops for event col ('2e17' or 'last_change'): the log-rank test
    plus the mean and Kaplan-Meier median time of each group and their differences (hi minus lo). Survival methods
    are used because a loop that never reaches the event only tells us the time is longer than its slots."""
    t = lambda g: g["t_" + col].to_numpy(float)
    e = lambda g: g["event_" + col].to_numpy(bool)
    out = logrank(t(hi), e(hi), t(lo), e(lo)) if len(hi) and len(lo) else dict(z=math.nan, p_two=math.nan, p_a_slower=math.nan)
    out.update(mean_hi=float(t(hi).mean()) if len(hi) else math.nan, mean_lo=float(t(lo).mean()) if len(lo) else math.nan,
               median_hi=km_median(t(hi), e(hi)), median_lo=km_median(t(lo), e(lo)),
               events_hi=int(e(hi).sum()), events_lo=int(e(lo).sum()), n_hi=len(hi), n_lo=len(lo))
    out["diff"] = out["mean_hi"] - out["mean_lo"]
    out["median_diff"] = out["median_hi"] - out["median_lo"]    # inf - inf is NaN: not evaluable
    return out


def effect(df, factor, hi_level, lo_level, seed):
    """hi minus lo for one main effect over the non-harm loops: stall, the two times, log dollars, sealed gain."""
    hi, lo = group(df, factor, hi_level), group(df, factor, lo_level)
    sh, sl = hi.stall.dropna(), lo.stall.dropna()
    res = dict(factor=factor, hi=hi_level, lo=lo_level, n_hi=len(hi), n_lo=len(lo))
    res["stall"] = dict(rate_hi=float(sh.mean()) if len(sh) else math.nan, rate_lo=float(sl.mean()) if len(sl) else math.nan,
                        k_hi=int(sh.sum()), n_hi=len(sh), k_lo=int(sl.sum()), n_lo=len(sl),
                        p=fisher(int(sh.sum()), len(sh), int(sl.sum()), len(sl)))
    res["stall"]["diff"] = res["stall"]["rate_hi"] - res["stall"]["rate_lo"]
    res["time_2e17"] = survival(hi, lo, "2e17")
    res["time_last_change"] = survival(hi, lo, "last_change")
    # dollars are compared on the log scale, so the estimate is a log ratio and exp() of it a geometric cost ratio
    logs = lambda g, col: np.log(np.maximum(g[col].to_numpy(float), 1e-9))
    cost_cols = ["model_usd"] + ["total_usd_gpu_%s" % str(p).replace(".", "p") for p in CONSTANTS["gpu_prices"]]
    res["log_cost"] = {}
    for col in cost_cols:
        r = perm(logs(lo, col), logs(hi, col), seed) if len(hi) and len(lo) else dict(estimate=math.nan, p=math.nan)
        res["log_cost"][col] = dict(log_diff=r["estimate"], ratio=math.exp(r["estimate"]) if r["estimate"] == r["estimate"] else math.nan,
                                    p=r["p"])
    res["sealed_gain"] = {}
    for col in ("sealed_gain_global", "sealed_gain_paired"):
        x, y = lo[col].dropna().to_numpy(float), hi[col].dropna().to_numpy(float)
        r = perm(x, y, seed, interval=True) if len(x) and len(y) else dict(estimate=math.nan, p=math.nan, lo=math.nan, hi=math.nan)
        res["sealed_gain"][col] = dict(diff=r["estimate"], p=r["p"], lo=r["lo"], hi=r["hi"], n_hi=len(y), n_lo=len(x))
    return res


def harm_control(df):
    """H against SS, the positive control. H differs from SS only in that half of the coder's edits are thrown
    away, so a sensitive measure should show it slower to 2**17. p_h_slower (one-sided log-rank) is what the KILL rule reads; the stall
    comparison and the no-op counts are reported next to it."""
    h, ss = df[df.cell == "H"], df[df.cell == "SS"]
    out = dict(n_h=len(h), n_ss=len(ss), noops_total=int(h.harm_noops.sum()) if len(h) else 0,
               noops_per_loop=float(h.harm_noops.mean()) if len(h) else math.nan)
    out["time_2e17"] = survival(h, ss, "2e17")
    out["p_h_slower"] = out["time_2e17"]["p_a_slower"]
    sh, ssd = h.stall.dropna(), ss.stall.dropna()
    out["stall"] = dict(rate_h=float(sh.mean()) if len(sh) else math.nan, rate_ss=float(ssd.mean()) if len(ssd) else math.nan,
                        p_two=fisher(int(sh.sum()), len(sh), int(ssd.sum()), len(ssd)),
                        p_h_more=fisher(int(sh.sum()), len(sh), int(ssd.sum()), len(ssd), "greater"))
    return out


def pooled_per_change(df, cell, col="model_usd"):
    """Dollars (column col) of a cell's loops divided by their backed accepted changes."""
    g = df[df.cell == cell]
    k = g.accepts_backed.sum()
    return float(g[col].sum() / k) if len(g) and k > 0 else math.nan


def cost_ratio(df):
    """Pooled model dollars per backed accepted change, SS over QT (the PASS rule wants QT at least 5 times cheaper),
    and the same ratio with GPU time added at each price of the grid (informational)."""
    ratio = lambda col: pooled_per_change(df, "SS", col) / pooled_per_change(df, "QT", col)
    r = dict(usd_per_change_ss=pooled_per_change(df, "SS"), usd_per_change_qt=pooled_per_change(df, "QT"))
    ok = bool(r["usd_per_change_qt"])   # NaN (no backed change) divides to NaN, zero would raise
    r["ratio"] = ratio("model_usd") if ok else math.nan
    r["ratio_with_gpu"] = {str(p): ratio("total_usd_gpu_%s" % str(p).replace(".", "p")) if ok else math.nan
                           for p in CONSTANTS["gpu_prices"]}
    return r


def decide(harm_p, planner, coder, ratio, c=CONSTANTS):
    """The PASS, KILL or PIVOT block of P2_PREREG. planner and coder are dicts with stall_diff and slots_diff (Q or T
    minus S; slots_diff is the difference of the Kaplan-Meier medians of slots to 2^17); NaN means not evaluable, which
    counts as 'not shown' for every condition. Precedence: KILL, PIVOT, PASS."""
    # v == v is False only for NaN, so a value that could not be computed satisfies neither comparison
    lt = lambda v, m: bool(v == v and v < m)
    gt = lambda v, m: bool(v == v and v > m)
    cond = dict(
        harm_detected=lt(harm_p, c["harm_alpha"]),
        planner_stall_ok=lt(planner["stall_diff"], c["stall_margin"]), coder_stall_ok=lt(coder["stall_diff"], c["stall_margin"]),
        planner_slots_ok=lt(planner["slots_diff"], c["slots_margin"]), coder_slots_ok=lt(coder["slots_diff"], c["slots_margin"]),
        qt_cheaper=bool(ratio == ratio and ratio >= c["cost_ratio_min"]))
    hurts = lambda e: gt(e["stall_diff"], c["stall_margin"]) or gt(e["slots_diff"], c["slots_margin"])
    holds = lambda e: lt(e["stall_diff"], c["stall_margin"]) and lt(e["slots_diff"], c["slots_margin"])
    label = lambda e: "cheap tier hurts" if hurts(e) else "cheap tier holds" if holds(e) else "unclear"
    roles = dict(planner=label(planner), coder=label(coder))
    pivot = hurts(planner)
    coder_hurts = hurts(coder)
    kill = not cond["harm_detected"]   # harm control not detected: the factorial is then only a cost breakdown
    passed = all(cond.values())
    verdict = ("KILL" if kill else "PIVOT" if pivot else "CODER HURTS" if coder_hurts else "PASS" if passed
               else "NO VERDICT")
    return dict(verdict=verdict, kill=kill, pivot=pivot, coder_hurts=coder_hurts, passed=passed, conditions=cond,
                roles=roles)


def quick_decision(df):
    """decide() on the loops of df, computed without the permutation tests. It gives the verdict of the run and, on the
    loops without a load-flagged job, the sensitivity verdict."""
    def main_effect(factor, hi_level, lo_level):
        hi, lo = group(df, factor, hi_level), group(df, factor, lo_level)
        sh, sl = hi.stall.dropna(), lo.stall.dropna()
        return dict(stall_diff=float(sh.mean() - sl.mean()) if len(sh) and len(sl) else math.nan,
                    slots_diff=survival(hi, lo, "2e17")["median_diff"])
    harm, ratio = harm_control(df), cost_ratio(df)
    return decide(harm["p_h_slower"], main_effect("planner", "Q", "S"), main_effect("coder", "T", "S"), ratio["ratio"])


# ------------------------------------------------------------------ per GPU
def gpu_breakdown(gpu_rows, run):
    """Per-GPU job counts, load flags and wall time, plus the baseline sentinel runs of each GPU and a one-way
    ANOVA across GPUs on them. GPU offsets are fixed at zero in P2, so this is the check that no GPU scores
    differently from the others."""
    sent = A.read_jsonl(run / "launch_sentinels.jsonl") + A.read_jsonl(run / "sentinels.jsonl")
    s = pd.DataFrame([r for r in sent if r.get("status") == "ok" and r.get("official_val_bpb") is not None and r.get("gpu") is not None])
    out = {}
    jobs = gpu_rows.dropna(subset=["gpu"])
    gpus = sorted(set(jobs.gpu.astype(int)) | (set(s.gpu.astype(int)) if len(s) else set()))
    for g in gpus:
        j = jobs[jobs.gpu.astype(int) == g] if len(jobs) else jobs
        sv = s[s.gpu.astype(int) == g].official_val_bpb.astype(float) if len(s) else pd.Series(dtype=float)
        out[str(g)] = dict(loop_jobs=len(j), load_flagged_jobs=int(j.flagged.sum()) if len(j) else 0,
                           wall_s_mean=float(j.wall_s.mean()) if len(j) else math.nan,
                           jobs_by_cell={c: int((j.cell == c).sum()) for c in CELLS} if len(j) else {},
                           sentinels=len(sv), sentinel_mean=float(sv.mean()) if len(sv) else math.nan,
                           sentinel_sd=float(sv.std(ddof=1)) if len(sv) > 1 else math.nan)
    groups = [s[s.gpu.astype(int) == g].official_val_bpb.astype(float).to_numpy() for g in gpus] if len(s) else []
    groups = [x for x in groups if len(x) >= 2]
    anova_p = float(stats.f_oneway(*groups)[1]) if len(groups) >= 2 else math.nan
    return dict(per_gpu=out, sentinel_anova_p=anova_p, n_sentinels=len(s))


# ------------------------------------------------------------------ report
def f(v, nd=3):
    """A number for the report with nd decimals; n/a for None or NaN, inf and -inf spelled out."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return ("inf" if v > 0 else "-inf") if isinstance(v, float) and math.isinf(v) else ("%.*f" % (nd, v))


def fmed(v):
    """A Kaplan-Meier median for the report: 'not reached' when more than half of the group was censored."""
    return "not reached" if isinstance(v, float) and math.isinf(v) else f(v, 1)


def table_md(header, rows):
    """Markdown table from a header list and rows of ready-made strings."""
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def build_report(df, res, skipped):
    """report.md as one string: the verdict with its six conditions first, then the cell table, main effects, cost,
    sealed gain, harm control, served fractions, per-GPU table and the load-flag sensitivity verdict."""
    L = ["# P2 analysis: which loop role earns the expensive model", ""]
    d = res["decision"]
    L += ["## Verdict: %s" % d["verdict"], ""]
    names = {"harm_detected": "harm control detected (H slower than SS, one-sided log-rank p < %s)" % CONSTANTS["harm_alpha"],
             "planner_stall_ok": "planner main effect on stall under +%s" % CONSTANTS["stall_margin"],
             "coder_stall_ok": "coder main effect on stall under +%s" % CONSTANTS["stall_margin"],
             "planner_slots_ok": "planner main effect on median slots to 2^17 under +%s slots" % CONSTANTS["slots_margin"],
             "coder_slots_ok": "coder main effect on median slots to 2^17 under +%s slots" % CONSTANTS["slots_margin"],
             "qt_cheaper": "QT at least %sx cheaper than SS per surviving change" % CONSTANTS["cost_ratio_min"]}
    L += [table_md(["PASS condition", "met"], [[names[k], "yes" if v else "no"] for k, v in d["conditions"].items()]), ""]
    L += ["- PASS (all six conditions): **%s**" % ("yes" if d["passed"] else "no"),
          "- KILL (the harm control is not detected): **%s**" % ("yes" if d["kill"] else "no"),
          "- PIVOT (the cheap planner raises stall by more than %s or delays 2^17 by more than %s slots): **%s**" % (
              CONSTANTS["stall_margin"], CONSTANTS["slots_margin"], "yes" if d["pivot"] else "no"),
          "- CODER HURTS (the cheap coder raises stall by more than %s or delays 2^17 by more than %s slots): **%s**" % (
              CONSTANTS["stall_margin"], CONSTANTS["slots_margin"], "yes" if d.get("coder_hurts") else "no"),
          "- Per role: planner **%s**, coder **%s**." % (d["roles"]["planner"], d["roles"]["coder"]),
          "- Precedence when more than one holds: KILL, then PIVOT, then CODER HURTS, then PASS. NO VERDICT means none applies.", ""]
    if d["kill"]:
        L += ["The harm control was not detected, so the factorial result stands only as a cost breakdown.", ""]
    if not (res["harm"]["n_h"] and res["harm"]["n_ss"]):
        L += ["The harm control could not be evaluated: this run has %d H and %d SS loops." % (
            res["harm"]["n_h"], res["harm"]["n_ss"]), ""]
    elif d["kill"]:
        L += ["Read the KILL with the power of the test in mind: with %d H against %d SS loops the one-sided log-rank test "
              "misses a real harm effect fairly often (pre-run simulation on the R1 times: power roughly 0.2 to 0.7 for 6 H "
              "loops, see the known limits in loop/design/P2_RUNBOOK.md). A non-detection here is weak evidence that the "
              "substrate is blind to a wasteful coder; compare the observed H and SS numbers in the harm control section." % (
                  res["harm"]["n_h"], res["harm"]["n_ss"]), ""]
    L += ["## Loops", ""]
    cells = []
    for c in CELLS:
        g = df[df.cell == c]
        if len(g):
            cells.append([c, str(len(g)), str(int((g.slots_completed < CONSTANTS["censor_slot"]).sum())), f(g.stall.mean(), 2),
                          f(g.t_2e17.mean(), 1), fmed(km_median(g.t_2e17.to_numpy(float), g.event_2e17.to_numpy(bool))),
                          f(g.t_last_change.mean(), 1), f(g.accepts.mean(), 1), f(g.accepts_backed.mean(), 1),
                          f(g.claude_usd.mean()), f(g.open_usd.mean(), 4), f(pooled_per_change(df, c), 3),
                          f(g.gpu_hours.mean(), 2), f(g.harm_noops.mean(), 1)])
    L += [table_md(["cell", "loops", "unfinished", "stall rate", "mean slots to 2^17", "median slots to 2^17",
                    "mean slots to last change", "accepts", "backed accepts", "Claude USD", "open USD",
                    "model USD per backed change", "GPU hours", "harm no-ops"], cells), ""]
    unfinished = df[df.slots_completed < CONSTANTS["censor_slot"]]
    L += ["Times are slots used (the slot index plus one for an event); a loop that never reaches the event is counted at its "
          "finished slots (20 for a finished loop) and marked censored in the log-rank test and the Kaplan-Meier median. "
          "%d of %d loops finished fewer than %d slots (stopped early or failed); the stall rate counts them with the final "
          "incumbent they had, so a cell with many unfinished loops overstates its stall." % (
              len(unfinished), len(df), CONSTANTS["censor_slot"])]
    if skipped:
        L += ["Not part of the experiment, left out: %s." % ", ".join(skipped)]
    L += [""]

    L += ["## Main effects (non-harm loops, hi minus lo)", ""]
    rows = []
    for key in ("planner", "coder"):
        e = res["effects"][key]
        s, t, lc = e["stall"], e["time_2e17"], e["time_last_change"]
        rows.append([("%s vs %s (%s)" % (e["hi"], e["lo"], key)), "%d vs %d" % (e["n_hi"], e["n_lo"]),
                     "%s vs %s: %s (p %s)" % (f(s["rate_hi"], 2), f(s["rate_lo"], 2), f(s["diff"], 2), f(s["p"], 3)),
                     "median %s vs %s: %s; mean %s vs %s: %s (log-rank p %s)" % (
                         fmed(t["median_hi"]), fmed(t["median_lo"]), f(t["median_diff"], 1), f(t["mean_hi"], 1),
                         f(t["mean_lo"], 1), f(t["diff"], 1), f(t["p_two"], 3)),
                     "%s vs %s: %s (p %s)" % (f(lc["mean_hi"], 1), f(lc["mean_lo"], 1), f(lc["diff"], 1), f(lc["p_two"], 3))])
    L += [table_md(["effect", "loops", "stall rate, difference (Fisher p)",
                    "slots to 2^17: median difference (decision rule), mean difference",
                    "mean slots to last change, difference (log-rank p)"], rows),
          "", "The PASS and PIVOT slot margins use the median difference (P2_PREREG); a median is 'not reached' when more than "
          "half of the group never got to 2^17, which counts as +inf for PIVOT and as not shown for PASS.", ""]
    L += ["Interaction (planner x coder) is descriptive: see the cell table above.", ""]

    L += ["## Cost", ""]
    rows = []
    for key in ("planner", "coder"):
        e = res["effects"][key]
        for col, v in e["log_cost"].items():
            rows.append(["%s %s vs %s" % (key, e["hi"], e["lo"]), COST_NAMES[col], f(v["ratio"], 3), f(v["p"], 4)])
    L += [table_md(["effect", "dollars", "ratio hi / lo (geometric)", "permutation p"], rows), ""]
    cr = res["cost_ratio"]
    L += ["Model dollars per backed accepted change, pooled: SS %s, QT %s, ratio SS / QT %s (PASS wants at least %s). Model dollars are "
          "Claude at list price plus open models at the price file." % (f(cr["usd_per_change_ss"], 4), f(cr["usd_per_change_qt"], 4),
                                                                       f(cr["ratio"], 2), CONSTANTS["cost_ratio_min"]),
          "With GPU time added the same ratio is %s (USD per A100-hour: ratio); GPU time is the same for every cell, so it shrinks "
          "the ratio toward 1." % ", ".join("%s: %s" % (p, f(v, 2)) for p, v in cr["ratio_with_gpu"].items()), ""]

    L += ["## Sealed held-out gain (secondary, two-sided permutation, %d percent interval)" % round(100 * CONSTANTS["ci_level"]), ""]
    rows = []
    for key in ("planner", "coder"):
        e = res["effects"][key]
        for col, v in e["sealed_gain"].items():
            rows.append(["%s %s vs %s" % (key, e["hi"], e["lo"]), "global anchor" if col.endswith("global") else "paired same wave",
                         "%d vs %d" % (v["n_hi"], v["n_lo"]), f(v["diff"], 5), "%s to %s" % (f(v["lo"], 5), f(v["hi"], 5)), f(v["p"], 3)])
    L += [table_md(["effect", "version", "loops", "difference", "interval", "p"], rows),
          "", "Global anchor: mean sealed of all usable anchor runs (%s) minus the loop's mean confirmation sealed. Paired: each "
          "confirmation run against the anchor of its own wave. Expected to be inconclusive." % f(res["anchor_sealed_mean"], 5), ""]

    h = res["harm"]
    L += ["## Harm control (H against SS)", ""]
    L += ["- Loops: %d H, %d SS. Replaced edits: %d in total, %s per H loop." % (h["n_h"], h["n_ss"], h["noops_total"], f(h["noops_per_loop"], 1)),
          "- Slots to 2^17, median H %s against SS %s, mean H %s against SS %s. One-sided log-rank p (H slower): **%s**." % (
              fmed(h["time_2e17"]["median_hi"]), fmed(h["time_2e17"]["median_lo"]), f(h["time_2e17"]["mean_hi"], 1),
              f(h["time_2e17"]["mean_lo"], 1), f(h["p_h_slower"], 4)),
          "- Stall rate H %s against SS %s; Fisher exact two-sided p %s, one-sided (H more) p %s." % (
              f(h["stall"]["rate_h"], 2), f(h["stall"]["rate_ss"], 2), f(h["stall"]["p_two"], 3), f(h["stall"]["p_h_more"], 3)), ""]

    L += ["## Served fractions and fallbacks", ""]
    rows = []
    for c in CELLS:
        g = df[df.cell == c]
        if len(g):
            rows.append([c, f(g.served_fraction_state.mean(), 3), f(g.served_fraction_ledger.mean(), 3),
                         f(g.planner_served_fraction.mean(), 3), str(int(g.n_fallbacks.sum())),
                         str(int((g.served_fraction_ledger < 0.8).sum())), str(int((g.planner_served_fraction < 1).sum()))])
    L += [table_md(["cell", "served fraction (state)", "served fraction (ledger)", "planner served fraction (ledger)",
                    "fallback events", "loops under 0.8", "loops with planner calls not served as designated"], rows),
          "", "The ledger column covers the coder, repairer, refresh, analyst and voter roles; the planner column covers the "
          "planner_backlog, planner_widen and planner_refresh calls, which carry the planner factor."]
    bad = df[df.planner_served_fraction < 1]
    if len(bad):
        L += ["Planner calls not all served by the designated model (check the fallbacks in state.json): %s." % ", ".join(
            "%s (%s)" % (r.loop_id, f(r.planner_served_fraction, 2)) for r in bad.itertuples())]
    L += [""]

    gb = res["gpu"]
    L += ["## Per GPU (GPU is a blocking factor)", ""]
    rows = [[g, str(v["loop_jobs"]), str(v["load_flagged_jobs"]), f(v["wall_s_mean"], 0),
             ", ".join("%s %d" % kv for kv in v["jobs_by_cell"].items() if kv[1]),
             str(v["sentinels"]), f(v["sentinel_mean"], 5), f(v["sentinel_sd"], 5)] for g, v in gb["per_gpu"].items()]
    L += [table_md(["GPU", "loop jobs", "load-flagged jobs", "mean wall s", "jobs by cell", "sentinels", "sentinel mean",
                    "sentinel sd"], rows),
          "", "One-way ANOVA across GPUs on %d sentinel runs: p %s." % (gb["n_sentinels"], f(gb["sentinel_anova_p"], 3)), ""]

    sens = res["load_flag_sensitivity"]
    L += ["## Load-flagged jobs (sensitivity)", ""]
    rows = []
    for c in CELLS:
        g = df[df.cell == c]
        if len(g):
            rows.append([c, str(int(g.n_load_flagged.sum())), str(int((g.n_load_flagged > 0).sum())), str(len(g))])
    L += [table_md(["cell", "load-flagged jobs", "loops with one", "loops"], rows), ""]
    if not sens["n_dropped"]:
        L += ["No job was load-flagged (no foreign process on the GPU before or after the run), so the verdict stands as is.", ""]
    elif sens["decision"] is None:
        L += ["Every loop has a load-flagged job: no sensitivity verdict.", ""]
    else:
        L += ["Without the %d loops that have a load-flagged job (%d loops left) the same rules give: **%s** (the run itself: %s). "
              "A difference means that a second process on a GPU may have moved the result." % (
                  sens["n_dropped"], sens["n_loops"], sens["decision"]["verdict"], d["verdict"]), ""]
    return "\n".join(L).replace(EM_DASH, "-") + "\n"


# ------------------------------------------------------------------ main
def analyze(run, prices, cell_map=None, seed=1):
    """The whole analysis of one run directory: (loops DataFrame, results dict, skipped loop ids). The verdict is
    computed twice, on all loops and on the loops without a load-flagged job (load_flag_sensitivity)."""
    df, gpu_rows, anchor_sealed, skipped = load_run(run, prices, cell_map)
    if df.empty:
        raise SystemExit("no loop of %s has a cell (plan.json cells, config cell, SS_01 style ids or --cell-map)" % run)
    eff = {"planner": effect(df, "planner", "Q", "S", seed), "coder": effect(df, "coder", "T", "S", seed)}
    harm, ratio = harm_control(df), cost_ratio(df)
    clean = df[df.n_load_flagged == 0]
    sensitivity = dict(n_loops=len(clean), n_dropped=len(df) - len(clean),
                       decision=quick_decision(clean) if len(clean) else None)
    res = dict(run=str(run), n_loops=len(df), effects=eff, harm=harm, cost_ratio=ratio, anchor_sealed_mean=anchor_sealed,
               gpu=gpu_breakdown(gpu_rows, Path(run)), constants=CONSTANTS, decision=quick_decision(df),
               load_flag_sensitivity=sensitivity)
    return df, res, skipped


def parse_cell_map(text):
    """'C=SS,D=ST' as {'C': 'SS', 'D': 'ST'}; exits on a value that is not a cell."""
    out = {}
    for part in filter(None, (text or "").split(",")):
        k, _, v = part.partition("=")
        if v not in CELLS:
            raise SystemExit("--cell-map: %r is not one of %s" % (v, ", ".join(CELLS)))
        out[k.strip()] = v
    return out


def main(argv=None):
    """Command line entry point: writes loops.csv, results.json and report.md, prints the verdict."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prices", default=str(DEFAULT_PRICES))
    ap.add_argument("--cell-map", default=None, help="loop id prefix to cell, e.g. C=SS,D=ST (smoke test on an older run)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    prices = json.loads(Path(a.prices).read_text())
    df, res, skipped = analyze(a.run, prices, parse_cell_map(a.cell_map), a.seed)
    out = Path(a.out or Path(a.run) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "loops.csv", index=False)
    (out / "results.json").write_text(json.dumps(A.clean_json(res), indent=1).replace(EM_DASH, "-"))
    (out / "report.md").write_text(build_report(df, res, skipped))
    print("verdict %s; %d loops; wrote %s" % (res["decision"]["verdict"], len(df), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
