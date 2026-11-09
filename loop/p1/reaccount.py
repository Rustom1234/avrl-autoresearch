"""Project 1: zero-GPU, zero-API re-accounting of run R1 (loop/runs/R1).

Reads only files already on disk. Computes, per loop and per arm: cost reconciliation, cost to plateau, cost to reach
a batch size of 2**17, the stall flag, halting rules, a noise replay of the converged regime, a frontier of
acceptance rules by simulation, and a GPU ANOVA on the sentinels. See loop/p1/README.md and
loop/design/P1_PREREG.md.

    .venv/bin/python loop/p1/reaccount.py --half burnin

The 22 loops are split inside each arm: sort the loop ids, even positions are burn-in, odd positions are test.
--half test or --half all is refused unless the environment variable P1_ALLOW_TEST=1 is set, and the test loops are
never opened when --half burnin is used.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "loop" / "core"))
import acceptance  # noqa: E402

RUN = ROOT / "loop" / "runs" / "R1"
PRICES_FILE = ROOT / "loop" / "models" / "open_prices.json"
BASE_TRAIN = ROOT / "loop" / "substrate" / "base" / "train_ampere.py"

TARGET_TBS = 2 ** 17          # the "converged" batch size
GPU_PRICES = (0.5, 1.0, 2.0, 4.0)   # USD per A100-hour
REF_PRICE = 2.0
PLATEAU_QUIET = 5             # quiet charged slots needed to call a plateau observed
HALT_KS = (3, 5, 8, 10)
HALT_CAPS = (5, 8, 10, 12)
LOSS_MARGIN = 0.004
KEEP_SHARE = 0.8
MAX_DOLLAR_FRAC = 0.60        # halting must spend no more than this share of a loop's dollars, on average
TRACKED = ("TOTAL_BATCH_SIZE", "DEPTH", "ASPECT_RATIO", "HEAD_DIM")
RULES = [("quiet", k) for k in HALT_KS] + [("after_first", k) for k in HALT_KS] + [("cap", n) for n in HALT_CAPS]
FRONTIER_RULES = ("keep_if_lower", "zband", "mean2", "median3")
FRONTIER_EFFECTS = (0.0, 0.001, 0.002, 0.004, 0.008)
FRONTIER_N_INC = 2            # chosen on burn-in (P1_PREREG.md); the test and all halves must not re-derive it
FROZEN_RULE = "after_first:5"  # halting rule selected on burn-in and frozen in P1_PREREG.md
FIXED_BURNIN = {"A_01", "A_03", "A_05", "A_07", "B_01", "B_03", "C_01", "C_03", "D_01", "D_03", "D_05", "D_07"}
FIXED_TEST = {"A_02", "A_04", "A_06", "A_08", "B_02", "C_02", "D_02", "D_04", "D_06", "D_08"}


# ---------------------------------------------------------------- split and guard
def split_loops(loop_ids):
    """Within each arm, sort the ids; even positions are burn-in, odd positions are test."""
    out = {}
    for arm in sorted({l[0] for l in loop_ids}):
        for i, l in enumerate(sorted(x for x in loop_ids if x[0] == arm)):
            out[l] = "burnin" if i % 2 == 0 else "test"
    return out


def check_half(half, env=None):
    """Refuse the test half (or all loops) unless P1_ALLOW_TEST=1 is set."""
    env = os.environ if env is None else env
    if half not in ("burnin", "test", "all"):
        sys.exit("--half must be burnin, test or all")
    if half != "burnin" and env.get("P1_ALLOW_TEST") != "1":
        sys.exit("refused: --half %s opens the test loops; set P1_ALLOW_TEST=1 only after the preregistration is frozen" % half)


def rule_label(rule):
    kind, p = rule
    return {"quiet": "quiet_k%d", "after_first": "after_first_k%d", "cap": "cap_n%d"}[kind] % p


def parse_rule(text):
    """'quiet:5' -> ('quiet', 5)."""
    kind, p = text.split(":")
    if (kind, int(p)) not in RULES:
        raise ValueError("unknown rule %r" % text)
    return kind, int(p)


# ---------------------------------------------------------------- reading train.py constants
def _ev(n):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -_ev(n.operand)
    if isinstance(n, ast.BinOp):
        a, b = _ev(n.left), _ev(n.right)
        ops = {ast.Pow: lambda: a ** b, ast.Mult: lambda: a * b, ast.Add: lambda: a + b, ast.Sub: lambda: a - b,
               ast.FloorDiv: lambda: a // b, ast.LShift: lambda: a << b}
        return ops[type(n.op)]()
    raise ValueError("unsupported constant expression: %s" % ast.dump(n))


def parse_consts(src, names=TRACKED):
    """Value of each tracked name from the LAST module-level assignment to it (later lines win)."""
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in names:
            out[n.targets[0].id] = _ev(n.value)
    return out


def check_consts(c, result):
    """Cross-check parsed DEPTH and TOTAL_BATCH_SIZE against what the run printed (skipped if nothing printed)."""
    pr = (result or {}).get("printed") or {}
    if pr.get("depth") is not None and pr["depth"] != c["DEPTH"]:
        raise ValueError("parsed DEPTH %s but the run printed %s" % (c["DEPTH"], pr["depth"]))
    if pr.get("num_steps") and pr.get("total_tokens_M"):
        tbs = pr["total_tokens_M"] * 1e6 / pr["num_steps"]
        if abs(tbs / c["TOTAL_BATCH_SIZE"] - 1) > 0.01:
            raise ValueError("parsed TOTAL_BATCH_SIZE %s but the run implies %.0f" % (c["TOTAL_BATCH_SIZE"], tbs))


# ---------------------------------------------------------------- ledger readers
def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def groups(recs):
    """Experiments: a new group starts at kind candidate or crash_fix (same rule as loop/core/ledger.py)."""
    out = []
    for r in recs:
        if r["decision"] == "infra":
            continue
        if r["kind"] in ("candidate", "crash_fix") or not out:
            out.append([r])
        else:
            out[-1].append(r)
    return out


def change_events(recs):
    """One entry per accepted experiment (an incumbent change): the accept row's slot, the slot of the first run,
    the raw official mean of the runs behind it, and how many runs that is."""
    out = []
    for g in groups(recs):
        if g[-1]["decision"] != "accept":
            continue
        vals = [r["official_val_bpb"] for r in g if r["job_id"] and r["official_val_bpb"] is not None]
        out.append({"slot": g[-1]["slot"], "first_slot": g[0]["slot"], "official": fmean(vals), "n_runs": len(vals)})
    return out


def call_dollars(call, per_m):
    """(Claude dollars, open-model dollars) of one ledger model call. Claude: cost_usd_plan. Open models: per-token
    hosted prices from open_prices.json (their cost_usd_plan is 0)."""
    p = per_m.get(call["served_model"])
    if p is None:
        return call["cost_usd_plan"], 0.0
    return 0.0, (call["tokens_in"] * p["input"] + call["tokens_out"] * p["output"]) / 1e6


def slot_costs(recs, per_m, n_slots, tail_calls=()):
    """Per-slot Claude and open-model dollars. Every row counts (charged or not); calls ride on the row after
    them. tail_calls (state.pending_calls: the analyst call after the last slot) go on the last slot."""
    claude, opn = [0.0] * n_slots, [0.0] * n_slots
    for r in recs:
        s = min(r["slot"], n_slots - 1)
        for c in r["model_calls"]:
            a, b = call_dollars(c, per_m)
            claude[s] += a
            opn[s] += b
    for c in tail_calls:
        a, b = call_dollars(c, per_m)
        claude[-1] += a
        opn[-1] += b
    return claude, opn


def gpu_seconds(loop_dir, recs, n_slots):
    """Per-slot measured wall seconds from slots/<slot>/result.json, one job per slot, job ids must match."""
    out = [0.0] * n_slots
    seen = set()
    for r in recs:
        if not r["job_id"]:
            continue
        s = r["slot"]
        if s in seen:
            raise ValueError("two jobs in slot %d of %s" % (s, loop_dir))
        seen.add(s)
        res = json.loads((Path(loop_dir) / "slots" / str(s) / "result.json").read_text())
        if res["job_id"] != r["job_id"]:
            raise ValueError("job id mismatch in %s slot %d" % (loop_dir, s))
        out[s] = float(res["wall_seconds"])
    return out


def build_tracks(changes, n_slots, base):
    """Incumbent after each slot: raw official mean, number of runs behind it, batch size. base = the start state."""
    off, n, tbs = [], [], []
    cur = dict(base)
    by_slot = {c["slot"]: c for c in changes}
    for s in range(n_slots):
        if s in by_slot:
            c = by_slot[s]
            cur = {"official": c["official"], "n": c["n"], "tbs": c["tbs"]}
        off.append(cur["official"])
        n.append(cur["n"])
        tbs.append(cur["tbs"])
    return off, n, tbs


def load_loop(run, lid, per_m, base):
    """Everything the analysis needs about one loop, as plain lists indexed by slot."""
    d = Path(run) / lid
    recs = jl(d / "ledger.jsonl")
    state = json.loads((d / "state.json").read_text())
    n = state["n_slots"]
    if sorted(r["slot"] for r in recs if r["charged"]) != list(range(n)):
        raise ValueError("%s: charged slots are not 0..%d" % (lid, n - 1))
    jobs = [0] * n
    for r in recs:
        if r["job_id"]:
            jobs[r["slot"]] += 1
    claude, opn = slot_costs(recs, per_m, n, state.get("pending_calls") or [])
    changes = []
    for ev in change_events(recs):
        sd = d / "slots" / str(ev["slot"])
        c = parse_consts((sd / "train.py").read_text())
        check_consts(c, json.loads((sd / "result.json").read_text()))
        changes.append({"slot": ev["slot"], "official": ev["official"], "n": ev["n_runs"], "tbs": c["TOTAL_BATCH_SIZE"]})
    if not changes:
        raise ValueError("%s: no incumbent change, plateau is undefined" % lid)
    inc_off, inc_n, inc_tbs = build_tracks(changes, n, base)
    final = parse_consts((d / "incumbent.py").read_text())
    if final["TOTAL_BATCH_SIZE"] != inc_tbs[-1]:
        raise ValueError("%s: incumbent.py batch size differs from the last accepted slot" % lid)
    rows = [{"slot": r["slot"], "kind": r["kind"], "decision": r["decision"], "official": r["official_val_bpb"]}
            for r in recs if r["job_id"] and r["official_val_bpb"] is not None]
    return {"id": lid, "arm": lid[0], "n_slots": n, "jobs": jobs, "gpu_s": gpu_seconds(d, recs, n), "claude": claude,
            "open": opn, "change_slots": sorted(c["slot"] for c in changes), "inc_official": inc_off, "inc_n": inc_n,
            "inc_tbs": inc_tbs, "final_tbs": final["TOTAL_BATCH_SIZE"], "final_depth": final["DEPTH"], "rows": rows}


# ---------------------------------------------------------------- cost math
def cost_upto(L, last, price):
    """Jobs, GPU hours and dollars through slot `last` inclusive (capped at the final slot)."""
    last = min(last, L["n_slots"] - 1)
    gpu_h = sum(L["gpu_s"][:last + 1]) / 3600.0
    claude, opn = sum(L["claude"][:last + 1]), sum(L["open"][:last + 1])
    return {"jobs": sum(L["jobs"][:last + 1]), "gpu_h": gpu_h, "claude": claude, "open": opn, "model": claude + opn,
            "total": claude + opn + gpu_h * price}


def loop_total(L, price):
    return cost_upto(L, L["n_slots"] - 1, price)


def crossover_price(model_x, gpu_h_x, model_y, gpu_h_y):
    """GPU price p where model_x + gpu_h_x * p equals model_y + gpu_h_y * p. None if the lines are parallel."""
    if gpu_h_x == gpu_h_y:
        return None
    return (model_y - model_x) / (gpu_h_x - gpu_h_y)


def crossover(model_x, gpu_h_x, model_y, gpu_h_y):
    """crossover_price kept only if positive, plus the cost gap x minus y at GPU prices 0 and 4 (to say who is cheaper)."""
    p = crossover_price(model_x, gpu_h_x, model_y, gpu_h_y)
    return {"price": p if p is not None and p > 0 else None, "raw_price": p,
            "x_minus_y_at_0": model_x - model_y, "x_minus_y_at_4": (model_x + 4 * gpu_h_x) - (model_y + 4 * gpu_h_y)}


def mean_of(rows, key):
    return fmean(r[key] for r in rows)


# ---------------------------------------------------------------- plateau, reach, stall
def plateau(L, quiet=PLATEAU_QUIET):
    """Slot of the last incumbent change. Observed if at least `quiet` charged slots follow it with no change,
    otherwise right-censored at n_slots."""
    last = max(L["change_slots"])
    observed = L["n_slots"] - 1 - last >= quiet
    return {"last_change": last, "slot": last if observed else L["n_slots"], "censored": not observed}


def reach(L, thr=TARGET_TBS):
    """First slot whose incumbent has TOTAL_BATCH_SIZE <= thr; censored at n_slots if it never does."""
    s = next((i for i, t in enumerate(L["inc_tbs"]) if t <= thr), None)
    return {"slot": L["n_slots"] if s is None else s, "censored": s is None}


def km_curve(times, events):
    """Kaplan-Meier survival at each event time. times and events are parallel lists (event False = censored)."""
    out, surv = [], 1.0
    for t in sorted({t for t, e in zip(times, events) if e}):
        at_risk = sum(1 for x in times if x >= t)
        died = sum(1 for x, e in zip(times, events) if e and x == t)
        surv *= 1 - died / at_risk
        out.append((t, surv))
    return out


def km_median(times, events):
    """Smallest event time where the survival curve is at or below 0.5; None if it never gets there."""
    return next((t for t, s in km_curve(times, events) if s <= 0.5 + 1e-12), None)


def boot_diff(a, b, n_boot, rng, level=0.90):
    """Mean(a) minus mean(b) with a percentile bootstrap interval (resample each group with replacement)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1) - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- halting rules
def halt_slot(change_slots, n_slots, rule):
    """Slot after which the rule stops the loop; n_slots - 1 if it never fires.
    quiet k: k consecutive slots with no incumbent change (counting from slot 0).
    after_first k: the same, but counting starts after the first change.
    cap N: stop after N slots."""
    kind, p = rule
    cs = set(change_slots)
    if kind == "cap":
        return min(p - 1, n_slots - 1)
    if kind == "after_first" and not cs:
        return n_slots - 1
    first = min(cs) if kind == "after_first" else -1
    run = 0
    for s in range(first + 1, n_slots):
        run = 0 if s in cs else run + 1
        if run >= p:
            return s
    return n_slots - 1


def halting_outcome(L, rule, price=REF_PRICE):
    """Halting slot, fraction of the loop's dollars spent by then, and loss in raw official mean
    (incumbent at the halt minus final incumbent; positive means the halt left a worse incumbent)."""
    h = halt_slot(L["change_slots"], L["n_slots"], rule)
    frac = cost_upto(L, h, price)["total"] / loop_total(L, price)["total"]
    return {"halt": h, "frac": frac, "loss": L["inc_official"][h] - L["inc_official"][-1]}


def summarize_rule(outcomes):
    return {"n": len(outcomes), "share_within": fmean(1.0 if o["loss"] <= LOSS_MARGIN else 0.0 for o in outcomes),
            "mean_frac": fmean(o["frac"] for o in outcomes), "mean_loss": fmean(o["loss"] for o in outcomes)}


def select_rule(summaries):
    """Among rules keeping at least KEEP_SHARE of loops within the margin, the one with the lowest mean dollar
    fraction (ties go to the earlier rule in RULES). summaries maps rule label to summarize_rule output."""
    ok = [(s["mean_frac"], i, lab) for i, (lab, s) in enumerate(summaries.items()) if s["share_within"] >= KEEP_SHARE - 1e-9]
    return min(ok)[2] if ok else None


# ---------------------------------------------------------------- final sealed improvement
def final_improvements(conf_rows, loop_ids):
    """Per loop: paired improvement (same-wave anchor sealed minus run sealed, mean over the loop's confirmation
    runs) and global-anchor improvement (mean of all anchor sealed values minus the loop's mean sealed)."""
    ok = [r for r in conf_rows if r["status"] == "ok" and r["sealed_val_bpb"] is not None]
    anchors = [r for r in ok if r["kind"] == "anchor"]
    wave_anchor = {r["wave"]: r["sealed_val_bpb"] for r in anchors}
    g_mean = fmean(r["sealed_val_bpb"] for r in anchors)
    out = {}
    for l in loop_ids:
        runs = [r for r in ok if r["kind"] == "confirm" and r["loop_id"] == l]
        paired = [wave_anchor[r["wave"]] - r["sealed_val_bpb"] for r in runs if r["wave"] in wave_anchor]
        sealed = fmean(r["sealed_val_bpb"] for r in runs)
        out[l] = {"n_runs": len(runs), "sealed_mean": sealed, "paired": fmean(paired), "global": g_mean - sealed}
    return out


# ---------------------------------------------------------------- sentinels
def sentinel_rows(run):
    """(raw official value, gpu) of the 16 launch sentinels and the 36 live sentinels."""
    run = Path(run)
    rows = jl(run / "launch_sentinels.jsonl") + jl(run / "sentinels.jsonl")
    return [(r["official_val_bpb"], r["gpu"]) for r in rows if r["status"] == "ok" and r["official_val_bpb"] is not None]


def anova_gpu(rows):
    """One-way ANOVA of the sentinel values across GPUs."""
    by = {}
    for v, g in rows:
        by.setdefault(g, []).append(v)
    gs = sorted(by)
    f, p = stats.f_oneway(*[by[g] for g in gs])
    return {"F": float(f), "p": float(p), "df_between": len(gs) - 1, "df_within": len(rows) - len(gs),
            "per_gpu": {str(g): {"n": len(by[g]), "mean": float(np.mean(by[g])), "sd": float(np.std(by[g], ddof=1))} for g in gs}}


# ---------------------------------------------------------------- noise replay (converged regime, arm A)
def converged_view(L, thr=TARGET_TBS):
    """Observed converged-regime facts for one loop: candidates whose first run came after the incumbent reached thr.
    keep = the plain keep-if-lower test (first run strictly below the incumbent's raw mean) as R1 logged it
    (decision accept or audit on the candidate row). Returns None if the loop never reached thr."""
    r = reach(L, thr)
    if r["censored"]:
        return None
    cands, audits = [], []
    for row in L["rows"]:
        if row["slot"] <= r["slot"]:
            continue
        if row["kind"] == "candidate":
            bar = L["inc_official"][row["slot"] - 1]
            cands.append({"slot": row["slot"], "x": row["official"], "bar": bar,
                          "keep": row["decision"] in ("accept", "audit"), "below_bar": row["official"] < bar})
        elif row["kind"] == "audit":
            audits.append({"slot": row["slot"], "reject": row["decision"] == "reject"})
    return {"reach": r["slot"], "bar0": L["inc_official"][r["slot"]], "n0": L["inc_n"][r["slot"]],
            "slots_left": L["n_slots"] - 1 - r["slot"], "cands": cands, "audits": audits}


def observed_counts(views):
    """Totals over converged views, plus how often the logged keep decision agrees with 'first run below the bar'."""
    cands = [c for v in views for c in v["cands"]]
    audits = [a for v in views for a in v["audits"]]
    return {"n_loops": len(views), "candidates": len(cands), "keeps": sum(c["keep"] for c in cands),
            "audits": len(audits), "audit_rejects": sum(a["reject"] for a in audits),
            "audit_accepts": sum(not a["reject"] for a in audits),
            "bar_rule_agree": sum(c["keep"] == c["below_bar"] for c in cands),
            "first_run_mean": float(np.mean([c["x"] for c in cands])), "first_run_sd": float(np.std([c["x"] for c in cands], ddof=1)),
            "slots_left": [v["slots_left"] for v in views]}


def make_noise(kind, resid):
    """draw(rng, size, sd) -> noise. gaussian uses the per-sim sd; bootstrap resamples the sentinel residuals."""
    if kind == "gaussian":
        return lambda rng, size, sd=1.0: rng.standard_normal(size) * sd
    return lambda rng, size, sd=None: rng.choice(resid, size)


def sim_converged(starts, mu, sigma, n_sd, n_mu, sigma_u, kind, resid, n_sims, rng):
    """Pure-noise replay of R1's converged regime for arm A. starts = [(bar0, n0, slots_left)] per loop. Every candidate
    has true level mu (no effect). Each simulated slot is a candidate (keep if the run is below the bar) or, after a
    keep, the audit (a fresh run; accept if z over both runs >= 1.5, else reject). A keep in the last slot gets no
    audit. Parameter uncertainty: sd (estimated from n_sd runs) is drawn from its sampling law (gaussian), mu from
    its standard error (estimated from n_mu runs)."""
    draw = make_noise(kind, resid)
    if kind == "gaussian":
        sd = sigma * np.sqrt((n_sd - 1) / rng.chisquare(n_sd - 1, n_sims))
    else:
        sd = np.full(n_sims, sigma)
    mu_r = mu + sd / np.sqrt(n_mu) * rng.standard_normal(n_sims)
    tot = {k: np.zeros(n_sims, int) for k in ("candidates", "keeps", "audits", "audit_rejects", "audit_accepts")}
    for bar0, n0, left in starts:
        bar, n = np.full(n_sims, bar0), np.full(n_sims, float(n0))
        pend, x1 = np.zeros(n_sims, bool), np.zeros(n_sims)
        for _ in range(left):
            x = mu_r + draw(rng, n_sims, sd)
            z = (bar - (x1 + x) / 2) / (sigma_u * np.sqrt(0.5 + 1 / n))
            acc = pend & (z >= acceptance.Z_ACCEPT_REPLICATED)
            tot["audits"] += pend
            tot["audit_accepts"] += acc
            tot["audit_rejects"] += pend & ~acc
            bar = np.where(acc, (x1 + x) / 2, bar)
            n = np.where(acc, 2.0, n)
            cand = ~pend
            keep = cand & (x < bar)
            tot["candidates"] += cand
            tot["keeps"] += keep
            x1 = np.where(keep, x, x1)
            pend = keep
    return tot


def pred_interval(arr, level=0.90):
    """Integer-valued predictive interval (outward rounding), so 'inside' is a conservative statement."""
    a = round((1 - level) / 2, 12)
    return int(np.quantile(arr, a, method="lower")), int(np.quantile(arr, round(1 - a, 12), method="higher"))


def noise_replay(views, sent_vals, sigma_u, n_sims, rng):
    """Observed converged-regime counts versus the pure-noise predictions. Three nulls: Gaussian at the sentinel sd
    (the preregistered one), bootstrap of the sentinel residuals, and Gaussian at the sd of the converged first runs
    themselves (a sensitivity check, because the sentinels run the unmodified 2**19 baseline)."""
    obs = observed_counts(views)
    v = np.asarray(sent_vals, float)
    sigma, resid = float(v.std(ddof=1)), v - v.mean()
    mu = obs["first_run_mean"]
    starts = [(w["bar0"], w["n0"], w["slots_left"]) for w in views]
    out = {"observed": obs, "sentinel_sd": sigma, "sentinel_n": len(v), "mu_hat": mu,
           "first_run_sd_ratio": obs["first_run_sd"] / sigma, "predicted": {}}
    nulls = {"gaussian": (sigma, len(v), "gaussian"), "bootstrap": (sigma, len(v), "bootstrap"),
             "gaussian_conv": (obs["first_run_sd"], obs["candidates"], "gaussian")}
    for name, (sd, n_sd, kind) in nulls.items():
        sim = sim_converged(starts, mu, sd, n_sd, obs["candidates"], sigma_u, kind, resid, n_sims, rng)
        pr = {}
        for k in ("candidates", "keeps", "audits", "audit_rejects", "audit_accepts"):
            lo, hi = pred_interval(sim[k])
            pr[k] = {"mean": float(sim[k].mean()), "lo90": lo, "hi90": hi, "observed": obs[k],
                     "inside": lo <= obs[k] <= hi, "p_ge": float((sim[k] >= obs[k]).mean()),
                     "p_le": float((sim[k] <= obs[k]).mean())}
        out["predicted"][name] = pr
    return out


# ---------------------------------------------------------------- frontier of acceptance rules (simulation)
def sim_frontier(rule, delta, draw, sd, sigma_u, n_inc, n_sims, rng, n_cand=20):
    """n_cand candidate decisions against an incumbent measured by n_inc runs. Each candidate's true level is
    `delta` better than the current incumbent's true level. A keep moves the bar to the candidate's own statistic.
    One slot = one candidate decision. Returns keeps per n_cand candidates and mean runs per candidate (the whole
    stream, so later decisions face a bar already pulled down by earlier keeps), and the keep rate and runs of the
    first decision alone (against an incumbent not yet pulled down: the plain power of the rule)."""
    true = np.zeros(n_sims)
    bar = np.mean([draw(rng, n_sims, sd) for _ in range(n_inc)], axis=0)
    n = np.full(n_sims, float(n_inc))
    keeps, runs = np.zeros(n_sims), np.zeros(n_sims)
    first = None
    for _ in range(n_cand):
        tc = true - delta
        x1 = tc + draw(rng, n_sims, sd)
        if rule == "keep_if_lower":
            keep, stat, used, n_new = x1 < bar, x1, 1, 1.0
        elif rule == "zband":
            z1 = (bar - x1) / (sigma_u * np.sqrt(1 + 1 / n))
            sure = z1 >= acceptance.Z_ACCEPT
            mid = (z1 >= acceptance.Z_REJECT) & ~sure
            x2 = tc + draw(rng, n_sims, sd)
            z2 = (bar - (x1 + x2) / 2) / (sigma_u * np.sqrt(0.5 + 1 / n))
            keep = sure | (mid & (z2 >= acceptance.Z_ACCEPT_REPLICATED))
            stat = np.where(sure, x1, (x1 + x2) / 2)
            used = 1 + mid
            n_new = np.where(sure, 1.0, 2.0)
        elif rule == "mean2":
            stat = (x1 + tc + draw(rng, n_sims, sd)) / 2
            keep, used, n_new = stat < bar, 2, 2.0
        elif rule == "median3":
            stat = np.median([x1, tc + draw(rng, n_sims, sd), tc + draw(rng, n_sims, sd)], axis=0)
            keep, used, n_new = stat < bar, 3, 3.0
        else:
            raise ValueError(rule)
        keeps += keep
        runs += used
        if first is None:
            first = {"first_keep": float(keep.mean()), "first_runs": float(np.mean(used))}
        true = np.where(keep, tc, true)
        bar = np.where(keep, stat, bar)
        n = np.where(keep, n_new, n)
    return {"keeps": float(keeps.mean()), "runs_per_slot": float(runs.mean() / n_cand), **first}


def frontier(sent_vals, sigma_u, n_inc, n_sims, rng, conv_sd=None):
    """Every rule x injected effect x noise model. keeps at effect 0 are false keeps per 20 slots; first_keep at a
    positive effect is the true-keep rate of one decision. conv_sd adds a Gaussian at the converged-regime sd."""
    v = np.asarray(sent_vals, float)
    sd, resid = float(v.std(ddof=1)), v - v.mean()
    variants = {"gaussian": (make_noise("gaussian", None), sd), "bootstrap": (make_noise("bootstrap", resid), sd)}
    if conv_sd:
        variants["gaussian_conv"] = (make_noise("gaussian", None), conv_sd)
    return {name: {rule: {str(d): sim_frontier(rule, d, draw, s, sigma_u, n_inc, n_sims, rng) for d in FRONTIER_EFFECTS}
                   for rule in FRONTIER_RULES} for name, (draw, s) in variants.items()}


# ---------------------------------------------------------------- the whole analysis for one half
def slot_noise(values, n_per_group=24):
    """Spread of slot endpoints: sd, the standard error of a difference of two group means of n_per_group loops each, and the
    half-width of its 90 percent interval. Used to size the slot margin for P2."""
    sd = float(np.std(values, ddof=1))
    se = sd * (2.0 / n_per_group) ** 0.5
    return {"sd": sd, "se_diff": se, "half_width_90": 1.645 * se, "n": len(values)}


def by_arm(loops):
    out = {}
    for L in loops:
        out.setdefault(L["arm"], []).append(L)
    return dict(sorted(out.items()))


def analyse(loops, sent_rows, improvements, sigma_u, n_sims, n_boot, seed, frozen_rule=None, frozen_n_inc=None):
    arms = by_arm(loops)
    res = {"loops": [L["id"] for L in loops], "arms": {a: [L["id"] for L in ls] for a, ls in arms.items()}}

    # 1. cost reconciliation
    per_loop = {}
    for L in loops:
        tot = loop_total(L, REF_PRICE)
        acc = len(L["change_slots"])
        per_loop[L["id"]] = {"arm": L["arm"], "jobs": tot["jobs"], "gpu_h": tot["gpu_h"], "claude": tot["claude"],
                             "open": tot["open"], "accepts": acc,
                             "total": {str(p): loop_total(L, p)["total"] for p in GPU_PRICES},
                             "per_accept": {str(p): loop_total(L, p)["total"] / acc for p in GPU_PRICES}}
    per_arm = {}
    for a, ls in arms.items():
        rows = [per_loop[L["id"]] for L in ls]
        acc = sum(r["accepts"] for r in rows)
        per_arm[a] = {"n_loops": len(ls), "jobs": mean_of(rows, "jobs"), "gpu_h": mean_of(rows, "gpu_h"),
                      "claude": mean_of(rows, "claude"), "open": mean_of(rows, "open"), "accepts": acc / len(ls),
                      "total": {str(p): fmean(r["total"][str(p)] for r in rows) for p in GPU_PRICES},
                      "per_accept": {str(p): sum(r["total"][str(p)] for r in rows) / acc for p in GPU_PRICES}}
    cross = {}
    for ref in ("A", "C"):
        if "D" in per_arm and ref in per_arm:
            x, y = per_arm["D"], per_arm[ref]
            cross["D_vs_" + ref] = crossover(x["claude"] + x["open"], x["gpu_h"], y["claude"] + y["open"], y["gpu_h"])
    res["cost"] = {"per_loop": per_loop, "per_arm": per_arm, "crossover_full_loop": cross}

    # 2. plateau
    pl = {}
    for L in loops:
        p = plateau(L)
        c = cost_upto(L, p["slot"], REF_PRICE)
        pl[L["id"]] = {**p, "jobs": c["jobs"], "gpu_h": c["gpu_h"], "model": c["model"], "total": c["total"],
                       "claude": c["claude"], "open": c["open"]}
    pl_arm, pl_cross = {}, {}
    for a, ls in arms.items():
        rows = [pl[L["id"]] for L in ls]
        t, e = [r["slot"] for r in rows], [not r["censored"] for r in rows]
        pl_arm[a] = {"n_loops": len(rows), "n_censored": sum(r["censored"] for r in rows), "mean_slot": fmean(t),
                     "km_median": km_median(t, e), "jobs": mean_of(rows, "jobs"), "gpu_h": mean_of(rows, "gpu_h"),
                     "model": mean_of(rows, "model"), "total": mean_of(rows, "total")}
    for ref in ("A", "C"):
        if "D" in pl_arm and ref in pl_arm:
            x, y = pl_arm["D"], pl_arm[ref]
            pl_cross["D_vs_" + ref] = crossover(x["model"], x["gpu_h"], y["model"], y["gpu_h"])
    res["plateau"] = {"per_loop": pl, "per_arm": pl_arm, "crossover": pl_cross}

    # 3. reach 2**17
    rc = {}
    for L in loops:
        r = reach(L)
        c = cost_upto(L, r["slot"], REF_PRICE)
        rc[L["id"]] = {**r, "jobs": c["jobs"], "gpu_h": c["gpu_h"], "model": c["model"], "total": c["total"]}
    rc_arm = {}
    for a, ls in arms.items():
        rows = [rc[L["id"]] for L in ls]
        t, e = [r["slot"] for r in rows], [not r["censored"] for r in rows]
        rc_arm[a] = {"n_loops": len(rows), "n_censored": sum(r["censored"] for r in rows), "mean_slot": fmean(t),
                     "km_median": km_median(t, e), "jobs": mean_of(rows, "jobs"), "total": mean_of(rows, "total")}
    res["reach"] = {"per_loop": rc, "per_arm": rc_arm}

    # 4. stall
    st = {L["id"]: {"final_tbs": L["final_tbs"], "final_depth": L["final_depth"], "stalled": L["final_tbs"] > TARGET_TBS,
                    "draft_p2_flag": L["final_tbs"] > 2 ** 18 or L["final_depth"] > 6, **improvements[L["id"]]} for L in loops}
    grp = {}
    for key in ("paired", "global"):
        s = [v[key] for v in st.values() if v["stalled"]]
        n = [v[key] for v in st.values() if not v["stalled"]]
        grp[key] = {"stalled_mean": fmean(s) if s else None, "not_stalled_mean": fmean(n) if n else None,
                    "separated": bool(s and n and min(n) > max(s)), "stalled_max": max(s) if s else None,
                    "not_stalled_min": min(n) if n else None}
    flagged = [v["paired"] for v in st.values() if v["draft_p2_flag"]]
    unflagged = [v["paired"] for v in st.values() if not v["draft_p2_flag"]]
    res["stall"] = {"per_loop": st, "groups": grp, "n_stalled": sum(v["stalled"] for v in st.values()),
                    "draft_p2_flag": {"n_flagged": len(flagged), "flagged_mean_paired": fmean(flagged) if flagged else None,
                                      "unflagged_mean_paired": fmean(unflagged) if unflagged else None,
                                      "n_stalled_missed": sum(v["stalled"] and not v["draft_p2_flag"] for v in st.values())}}

    # 5. halting
    outcomes = {rule_label(r): {L["id"]: halting_outcome(L, r) for L in loops} for r in RULES}
    summ = {lab: summarize_rule(list(o.values())) for lab, o in outcomes.items()}
    sel = select_rule(summ)
    res["halting"] = {"outcomes": outcomes, "summary": summ, "selected": sel, "margin": LOSS_MARGIN, "keep_share": KEEP_SHARE}

    # 6. noise replay and frontier (sentinels and arm A loops only)
    rng = np.random.default_rng([seed, 6])
    vals = [v for v, _ in sent_rows]
    a_loops = arms.get("A", [])
    views = [w for w in (converged_view(L) for L in a_loops) if w]
    res["noise"] = {"replay": noise_replay(views, vals, sigma_u, n_sims, rng) if views else None}
    n_inc = frozen_n_inc or (int(np.median([w["n0"] for w in views])) if views else 2)
    res["noise"]["frontier_n_inc"] = n_inc
    conv_sd = res["noise"]["replay"]["observed"]["first_run_sd"] if views else None
    res["noise"]["frontier"] = frontier(vals, sigma_u, n_inc, n_sims, rng, conv_sd)

    # 7. GPU offsets
    res["anova"] = anova_gpu(sent_rows)

    # numbers handed to P2: how widely arm D's slot endpoints spread, and what that means for a 24 against 24 comparison
    d_ids = [L["id"] for L in arms.get("D", [])]
    if len(d_ids) > 1:
        res["handoff"] = {"plateau_slot_D": slot_noise([pl[i]["slot"] for i in d_ids]), "reach_slot_D": slot_noise([rc[i]["slot"] for i in d_ids])}

    # decision lines (a dry run on burn-in, the verdict on test)
    rng2 = np.random.default_rng([seed, 7])
    a_t = [pl[L["id"]] for L in arms.get("A", [])]
    d_t = [pl[L["id"]] for L in arms.get("D", [])]
    dec = {}
    if a_t and d_t:
        full = boot_diff([r["slot"] for r in a_t], [r["slot"] for r in d_t], n_boot, rng2)
        a_o, d_o = [r["slot"] for r in a_t if not r["censored"]], [r["slot"] for r in d_t if not r["censored"]]
        dec["claim1"] = {"censored_at_20": full, "excluding_censored": boot_diff(a_o, d_o, n_boot, rng2) if a_o and d_o else None,
                         "n_censored": {"A": sum(r["censored"] for r in a_t), "D": sum(r["censored"] for r in d_t)}}
    rule = frozen_rule or sel
    if rule:
        s = summ[rule]
        dec["halting"] = {"rule": rule, "share_within": s["share_within"], "mean_frac": s["mean_frac"],
                          "pass": bool(s["share_within"] >= KEEP_SHARE - 1e-9 and s["mean_frac"] <= MAX_DOLLAR_FRAC)}
    else:
        dec["halting"] = {"rule": None, "pass": False}
    rp = res["noise"]["replay"]
    if rp:
        g = rp["predicted"]["gaussian"]["audit_rejects"]
        dec["n2_test1"] = {"observed": g["observed"], "lo90": g["lo90"], "hi90": g["hi90"], "pass": g["inside"]}
    res["decisions"] = dec
    return res


# ---------------------------------------------------------------- report
def fm(x, nd=3):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def xdescribe(key, c):
    other = key[-1]
    d0, d4 = c["x_minus_y_at_0"], c["x_minus_y_at_4"]
    gap = "D minus %s: %+.2f USD at a GPU price of 0, %+.2f at 4" % (other, d0, d4)
    if c["price"] is not None:
        return "D and %s cost the same at %.3f USD per A100-hour (%s)" % (other, c["price"], gap)
    side = "cheaper" if max(d0, d4) < 0 else "dearer" if min(d0, d4) > 0 else "about equal"
    return "no crossing at a positive price, D is %s than %s at every price from 0 to 4 (%s)" % (side, other, gap)


def render_report(res, half, params):
    P = [str(p) for p in GPU_PRICES]
    o = ["# Project 1 re-accounting of R1: %s half" % half, "",
         "Zero GPU, zero API. Every number comes from the files in loop/runs/R1. Loops in this half: %s." % ", ".join(res["loops"]),
         "Dollars: Claude at cost_usd_plan, open models at the per-token prices in loop/models/open_prices.json, GPU at the "
         "price shown. GPU hours are measured wall seconds from slots/*/result.json. Slots are 0-based; a censored loop is "
         "recorded at slot 20.", "", "Parameters: %s." % json.dumps(params), ""]

    c = res["cost"]
    o += ["## 1. Cost reconciliation", "", "Mean per loop, by arm.", "",
          table(["arm", "loops", "GPU jobs", "GPU hours", "Claude USD", "open USD", "accepted changes"] + ["total @%s" % p for p in P],
                [[a, v["n_loops"], fm(v["jobs"], 1), fm(v["gpu_h"], 2), fm(v["claude"]), fm(v["open"]), fm(v["accepts"], 2)]
                 + [fm(v["total"][p], 2) for p in P] for a, v in c["per_arm"].items()]), "",
          "Dollars per accepted change (arm total dollars divided by arm accepted changes).", "",
          table(["arm"] + ["@%s" % p for p in P], [[a] + [fm(v["per_accept"][p], 2) for p in P] for a, v in c["per_arm"].items()]), "",
          "GPU price at which arm D's mean total per loop equals the other arm's (full 20-slot loop): "
          + "; ".join(xdescribe(k, v) for k, v in c["crossover_full_loop"].items()) + ".", "",
          "Per loop.", "",
          table(["loop", "jobs", "GPU h", "Claude USD", "open USD", "accepts"] + ["total @%s" % p for p in P],
                [[l, v["jobs"], fm(v["gpu_h"], 2), fm(v["claude"]), fm(v["open"]), v["accepts"]] + [fm(v["total"][p], 2) for p in P]
                 for l, v in c["per_loop"].items()]), ""]

    pl = res["plateau"]
    o += ["## 2. Plateau (slot of the last incumbent change)", "",
          "Observed if at least %d charged slots follow it with no change, otherwise censored at 20. Costs run up to and "
          "including the plateau slot (a censored loop is charged its whole run), GPU at %.0f USD per A100-hour." % (PLATEAU_QUIET, REF_PRICE), "",
          table(["loop", "last change", "plateau slot", "censored", "jobs", "GPU h", "model USD", "total USD"],
                [[l, v["last_change"], v["slot"], "yes" if v["censored"] else "no", v["jobs"], fm(v["gpu_h"], 2), fm(v["model"]), fm(v["total"], 2)]
                 for l, v in pl["per_loop"].items()]), "",
          table(["arm", "loops", "censored", "mean slot", "Kaplan-Meier median slot", "jobs", "GPU h", "model USD", "total USD"],
                [[a, v["n_loops"], v["n_censored"], fm(v["mean_slot"], 2), "not reached" if v["km_median"] is None else v["km_median"],
                  fm(v["jobs"], 1), fm(v["gpu_h"], 2), fm(v["model"]), fm(v["total"], 2)] for a, v in pl["per_arm"].items()]), "",
          "GPU price at which arm D's mean cost to plateau equals the other arm's: "
          + "; ".join(xdescribe(k, v) for k, v in pl["crossover"].items()) + ".", ""]

    rc = res["reach"]
    o += ["## 3. Reach 2^17 (first slot whose incumbent has TOTAL_BATCH_SIZE <= 131072)", "",
          table(["loop", "slot", "censored", "jobs", "GPU h", "model USD", "total USD"],
                [[l, v["slot"], "yes" if v["censored"] else "no", v["jobs"], fm(v["gpu_h"], 2), fm(v["model"]), fm(v["total"], 2)]
                 for l, v in rc["per_loop"].items()]), "",
          table(["arm", "loops", "censored", "mean slot", "Kaplan-Meier median slot", "jobs", "total USD"],
                [[a, v["n_loops"], v["n_censored"], fm(v["mean_slot"], 2), "not reached" if v["km_median"] is None else v["km_median"],
                  fm(v["jobs"], 1), fm(v["total"], 2)] for a, v in rc["per_arm"].items()]), ""]

    st = res["stall"]
    o += ["## 4. Stall (final incumbent TOTAL_BATCH_SIZE > 131072)", "",
          "Improvement = sealed val_bpb gain over the baseline: paired uses the same-wave anchor, global uses the mean of "
          "all 30 anchors. Higher is better.", "",
          table(["loop", "final batch size", "final depth", "stalled", "draft P2 flag", "paired improvement", "global-anchor improvement", "sealed mean"],
                [[l, v["final_tbs"], v["final_depth"], "yes" if v["stalled"] else "no", "yes" if v["draft_p2_flag"] else "no", fm(v["paired"], 4),
                  fm(v["global"], 4), fm(v["sealed_mean"], 4)] for l, v in st["per_loop"].items()]), "",
          table(["version", "stalled mean", "not stalled mean", "stalled max", "not stalled min", "groups separated"],
                [[k, fm(g["stalled_mean"], 4), fm(g["not_stalled_mean"], 4), fm(g["stalled_max"], 4), fm(g["not_stalled_min"], 4),
                  "yes" if g["separated"] else "no"] for k, g in st["groups"].items()]), "",
          "The draft P2 stall rule (final incumbent has batch size above 2^18 or depth above 6) flags %d loops; their mean paired "
          "improvement is %s against %s for the unflagged loops; it misses %d of the %d stalled loops." % (
              st["draft_p2_flag"]["n_flagged"], fm(st["draft_p2_flag"]["flagged_mean_paired"], 4),
              fm(st["draft_p2_flag"]["unflagged_mean_paired"], 4), st["draft_p2_flag"]["n_stalled_missed"], st["n_stalled"]), ""]

    h = res["halting"]
    o += ["## 5. Halting rules", "",
          "quiet k: stop after k consecutive slots with no incumbent change. after_first k: the same, counting starts after "
          "the first change. cap N: stop after N slots. Loss = raw official mean of the incumbent at the halt minus the "
          "final incumbent's (positive is worse). Within = loss <= %.3f. Dollar fraction at %.0f USD per A100-hour." % (LOSS_MARGIN, REF_PRICE), "",
          table(["rule", "share within", "mean dollar fraction", "mean loss"],
                [[k, fm(v["share_within"], 2), fm(v["mean_frac"], 3), fm(v["mean_loss"], 4)] for k, v in h["summary"].items()]), "",
          "Selected on this half (lowest mean dollar fraction among rules with share within >= %.0f%%): **%s**." % (KEEP_SHARE * 100, h["selected"] or "none qualifies"), ""]
    labs = list(h["outcomes"])
    ids = res["loops"]
    o += ["Halting slot and loss, per loop.", "",
          table(["loop"] + labs, [[l] + ["%d / %s" % (h["outcomes"][k][l]["halt"], fm(h["outcomes"][k][l]["loss"], 4)) for k in labs] for l in ids]), "",
          "Dollar fraction spent by the halt, per loop.", "",
          table(["loop"] + labs, [[l] + [fm(h["outcomes"][k][l]["frac"], 2) for k in labs] for l in ids]), ""]

    nz = res["noise"]
    rp = nz["replay"]
    o += ["## 6. Noise replay and acceptance-rule frontier (sentinels and arm A loops only)", ""]
    if rp:
        ob = rp["observed"]
        o += ["Noise sd from %d sentinel runs (raw official): %.5f. Converged regime = candidate first runs after the incumbent "
              "reached 2^17, in arm A loops %s. Observed: %d candidates, %d would-be keeps, %d audits, %d audit rejects, %d audit "
              "accepts. The logged keep decision matches 'first run strictly below the incumbent raw mean' for %d of %d candidates. "
              "Sd of the converged first runs is %.2f times the sentinel sd." % (
                  rp["sentinel_n"], rp["sentinel_sd"], res["arms"].get("A"), ob["candidates"], ob["keeps"], ob["audits"],
                  ob["audit_rejects"], ob["audit_accepts"], ob["bar_rule_agree"], ob["candidates"], rp["first_run_sd_ratio"]), "",
              "Pure-noise prediction (true effect zero for every candidate, level = mean converged first run %.4f, R1's rule, "
              "sigma 0.004743 in the audit z). Intervals are 90 percent predictive (include uncertainty in sd and level)." % rp["mu_hat"], ""]
        rows = []
        for kind, pr in rp["predicted"].items():
            for k, v in pr.items():
                rows.append([kind, k, v["observed"], fm(v["mean"], 2), "%d to %d" % (v["lo90"], v["hi90"]), "yes" if v["inside"] else "NO",
                             fm(v["p_ge"], 3)])
        o += [table(["noise", "count", "observed", "predicted mean", "90% interval", "inside", "P(sim >= observed)"], rows), ""]
    else:
        o += ["No arm A loop in this half reached 2^17, so no converged-regime replay.", ""]
    o += ["Frontier by simulation. One slot is one candidate decision; each loop has 20 of them against an incumbent measured by "
          "%d runs. Rules: keep_if_lower (first run below the incumbent mean), zband (R1's rule: reject z below 0.75, accept at 2.0 "
          "or above, replicate in between and accept at 1.5), mean2 (mean of 2 runs below the incumbent), median3 (median of 3 "
          "runs below the incumbent). A keep moves the incumbent to the candidate's own measurement. Noise: gaussian and bootstrap use the "
          "sentinel sd and residuals; gaussian_conv uses the sd of the converged first runs (sensitivity)." % nz["frontier_n_inc"], ""]
    for kind, fr in nz["frontier"].items():
        eff = list(map(str, FRONTIER_EFFECTS[1:]))
        o += ["Noise: %s. False keeps are per 20 slots with no true effect; the other columns are the chance one decision keeps a "
              "candidate that is truly better by that many bpb, with runs per decision in brackets." % kind, "",
              table(["rule", "runs/slot (no effect)", "false keeps / 20"] + ["true-keep @%s (runs)" % d for d in eff],
                    [[rule, fm(v["0.0"]["runs_per_slot"], 2), fm(v["0.0"]["keeps"], 2)]
                     + ["%s (%s)" % (fm(v[d]["first_keep"], 3), fm(v[d]["first_runs"], 2)) for d in eff]
                     for rule, v in fr.items()]), "",
              "Same, as the mean keep rate over a stream of 20 consecutive truly better candidates (later decisions face a bar "
              "already pulled down by earlier keeps).", "",
              table(["rule"] + ["stream keep rate @%s (runs/slot)" % d for d in eff],
                    [[rule] + ["%s (%s)" % (fm(v[d]["keeps"] / 20, 3), fm(v[d]["runs_per_slot"], 2)) for d in eff]
                     for rule, v in fr.items()]), ""]

    a = res["anova"]
    o += ["## 7. GPU offsets (one-way ANOVA on the sentinels)", "",
          "F(%d, %d) = %.3f, p = %.3f." % (a["df_between"], a["df_within"], a["F"], a["p"]), "",
          table(["GPU", "n", "mean", "sd"], [[g, v["n"], fm(v["mean"], 5), fm(v["sd"], 5)] for g, v in a["per_gpu"].items()]), ""]

    if "handoff" in res:
        hh = res["handoff"]
        o += ["## Spread of slot endpoints in arm D (for sizing the P2 slot margin)", "",
              "Sd of the slot (censored loops at 20), the standard error of a difference between two groups of 24 loops, and the "
              "half-width of its 90 percent interval.", "",
              table(["endpoint", "loops", "sd", "SE of 24 vs 24 difference", "90% half-width"],
                    [[k, v["n"], fm(v["sd"], 2), fm(v["se_diff"], 2), fm(v["half_width_90"], 2)] for k, v in hh.items()]), ""]

    d = res["decisions"]
    o += ["## Decision lines (%s)" % ("dry run on burn-in, not a verdict" if half == "burnin" else "verdict"), ""]
    if "claim1" in d:
        c1 = d["claim1"]
        f = c1["censored_at_20"]
        o += ["Claim 1: A minus D mean slots to plateau (censored at 20) = %.2f, 90%% bootstrap interval %.2f to %.2f, excludes zero: %s. "
              "Censored loops: A %d, D %d." % (f["diff"], f["lo"], f["hi"], f["excludes_zero"], c1["n_censored"]["A"], c1["n_censored"]["D"])]
        e = c1["excluding_censored"]
        o += ["Sensitivity excluding censored loops: %s." % ("not computable" if e is None else
              "diff %.2f, interval %.2f to %.2f, excludes zero: %s" % (e["diff"], e["lo"], e["hi"], e["excludes_zero"]))]
    hd = d["halting"]
    o += ["", "Halting: rule %s%s." % (hd["rule"], "" if hd["rule"] is None else ", share within %s, mean dollar fraction %s, pass (>= 80%% within at <= 60%% of dollars): %s" % (
        fm(hd["share_within"], 2), fm(hd["mean_frac"], 3), hd["pass"]))]
    if "n2_test1" in d:
        n2 = d["n2_test1"]
        o += ["", "N2 Test 1: observed converged-regime audit rejects %d, 90%% predictive interval %d to %d, inside: %s." % (
            n2["observed"], n2["lo90"], n2["hi90"], n2["pass"])]
    return "\n".join(o) + "\n"


# ---------------------------------------------------------------- main
def all_loop_ids(run):
    return sorted(p.name for p in Path(run).iterdir() if re.fullmatch(r"[A-Z]_\d+", p.name) and (p / "ledger.jsonl").exists())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--half", required=True, choices=["burnin", "test", "all"])
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--out", default=None, help="default: <run>/p1/<half>")
    ap.add_argument("--rule", default=None, help="frozen halting rule such as quiet:5 (required for test and all)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    check_half(a.half)
    if a.half != "burnin" and a.rule != FROZEN_RULE:
        sys.exit("give the halting rule frozen in P1_PREREG.md: --rule %s" % FROZEN_RULE)
    run = Path(a.run)
    split = split_loops(all_loop_ids(run))
    if a.run == str(RUN) and ({l for l, h in split.items() if h == "burnin"} != FIXED_BURNIN
                             or {l for l, h in split.items() if h == "test"} != FIXED_TEST):
        sys.exit("refused: the loop directories do not give the split fixed in P1_PREREG.md")
    chosen = [l for l, h in split.items() if a.half == "all" or h == a.half]
    per_m = json.loads(PRICES_FILE.read_text())["per_million_tokens"]
    anchor = json.loads((run / "anchor.json").read_text())
    base_c = parse_consts(BASE_TRAIN.read_text())
    base = {"official": fmean(v["val"] for v in anchor["values"]), "n": anchor["n"], "tbs": base_c["TOTAL_BATCH_SIZE"]}
    loops = [load_loop(run, l, per_m, base) for l in chosen]
    conf = jl(run / "confirmation" / "results.jsonl")
    imp = final_improvements(conf, chosen)
    params = {"half": a.half, "sims": a.sims, "boot": a.boot, "seed": a.seed, "sigma_used": anchor["sigma_used"],
              "base_official": base["official"], "base_tbs": base["tbs"], "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    res = analyse(loops, sentinel_rows(run), imp, anchor["sigma_used"], a.sims, a.boot, a.seed,
                  rule_label(parse_rule(a.rule)) if a.rule else None, FRONTIER_N_INC if a.half != "burnin" else None)
    res["params"] = params
    res["split"] = {l: split[l] for l in chosen}
    out = Path(a.out) if a.out else run / "p1" / a.half
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "report.md").write_text(render_report(res, a.half, params))
    print("wrote", out / "results.json", "and", out / "report.md")


if __name__ == "__main__":
    main()
