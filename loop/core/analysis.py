"""Analysis for the pre-registered routed-autoresearch experiment (DESIGN.md s12, PREREG.md).

    python -m loop.core.analysis --run <RUN> --out <RUN>/analysis [--seed 1] [--prereg PATH] [--calibration PATH]

Reads state.json and ledger.jsonl of every loop dir, confirmation/results.jsonl, sentinels.jsonl, anchor.json
and plan.json. Writes loop_runs.csv, results.json, report.md and six PNG figures. Assumptions and column
definitions: loop/core/ANALYSIS.md. Needs only numpy, pandas, matplotlib.
"""
import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ---- frozen constants (PREREG.md) ------------------------------------------------------------
NOISE_RESID_SD = 0.005857
SIGMA_USED = 1.15 * NOISE_RESID_SD
MIN_VALID_SLOTS = 12
DEGRADED_BELOW = 0.8
H1_ALPHA, H1_MARGIN = 0.05, 0.003
BH_Q = 0.10
CI_LEVEL = 0.90
SIGMA_TOL = 0.25
NONINF_MARGIN = 0.01
ARMS = ["A", "B", "C", "D"]
MAX_EXACT = 20000   # enumerate all splits up to this many (8 vs 8 = 12870), else Monte Carlo
N_MC = 20000
N_BOOT = 10000
CLOSED = ("sonnet", "haiku", "opus", "claude")   # served_model substrings that mean a Claude model
OPEN_ROLES = ("coder", "repairer", "planner_refresh", "analyst", "voter")   # roles the routing table can send to an open model
REPO = Path(__file__).resolve().parents[2]   # relative routing paths in loop configs are relative to this directory


# ---- io -------------------------------------------------------------------------------------
def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(p):
    out = []
    try:
        lines = Path(p).read_text().splitlines()
    except OSError:
        return out
    for ln in lines:
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass   # torn last line of a live run
    return out


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def num0(x):
    v = num(x)
    return 0.0 if math.isnan(v) else v


def finite(x):
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


# ---- statistics -----------------------------------------------------------------------------
def _splits(n, k, rng):
    """Rows of treatment indices (last k of n is the observed split). Exact if few enough, else Monte Carlo
    (observed split first, so p = (1 + hits) / (1 + N_MC), a valid permutation p-value)."""
    if math.comb(n, k) <= MAX_EXACT:
        return np.array(list(itertools.combinations(range(n), k))), "exact"
    mc = np.argsort(rng.random((N_MC, n)), axis=1)[:, :k]
    return np.vstack([np.arange(n - k, n), mc]), "monte-carlo"


def perm_test(x, y, rng, interval=False, alpha=1 - CI_LEVEL):
    """Two-sided permutation test of mean(y) - mean(x). With interval=True also the (1 - alpha) interval by
    test inversion under a constant additive-shift model: {delta : p(y - delta vs x) >= alpha}, where the
    shifted statistic of every permutation is T_w - delta * (c/k - (k - c)/(n - k)), c = overlap of the
    permuted treatment set with the true one. Unbounded ends are +-inf (tiny samples)."""
    x, y = finite(x), finite(y)
    nx, ny = len(x), len(y)
    out = dict(n_x=nx, n_y=ny, estimate=math.nan, p=math.nan, method="", n_perm=0, lo=math.nan, hi=math.nan)
    if nx < 1 or ny < 1:
        return out
    pooled = np.concatenate([x, y])
    n, tot = nx + ny, pooled.sum()
    t_obs = y.mean() - x.mean()
    idx, method = _splits(n, ny, rng)
    s = pooled[idx].sum(1)
    t0 = s / ny - (tot - s) / nx
    tol = 1e-12 + 1e-9 * abs(t_obs)
    out.update(estimate=t_obs, method=method, n_perm=len(idx), p=float((np.abs(t0) >= abs(t_obs) - tol).mean()))
    if interval and n > 2:
        c = (idx >= nx).sum(1)
        a = c / ny - (ny - c) / nx
        se = max(pooled.std(ddof=1) * math.sqrt(1 / nx + 1 / ny), 1e-6)
        grid = np.unique(np.concatenate([np.linspace(t_obs - 6 * se, t_obs + 6 * se, 401), [0.0, t_obs]]))
        pv = np.array([(np.abs(t0 - g * a) >= abs(t_obs - g) - tol).mean() for g in grid])
        keep = grid[pv >= alpha]
        if keep.size:
            out["lo"] = -math.inf if keep.min() <= grid[0] else float(keep.min())
            out["hi"] = math.inf if keep.max() >= grid[-1] else float(keep.max())
    return out


def boot_diff(x, y, rng, B=N_BOOT, level=CI_LEVEL):
    """Percentile bootstrap of mean(y) - mean(x): (estimate, lo, hi)."""
    x, y = finite(x), finite(y)
    if len(x) < 1 or len(y) < 1:
        return math.nan, math.nan, math.nan
    d = y[rng.integers(0, len(y), (B, len(y)))].mean(1) - x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    lo, hi = np.percentile(d, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(y.mean() - x.mean()), float(lo), float(hi)


def boot_mean(x, rng, B=N_BOOT, level=CI_LEVEL):
    x = finite(x)
    if len(x) < 1:
        return math.nan, math.nan, math.nan
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    lo, hi = np.percentile(m, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(x.mean()), float(lo), float(hi)


def bh(pvals, q=BH_Q):
    """Benjamini-Hochberg over exactly len(pvals) tests (NaN counts as p = 1 and still occupies a slot)."""
    p = np.array([1.0 if not np.isfinite(v) else v for v in pvals], dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1.0)
    qv = np.empty(m)
    qv[order] = adj
    return qv, qv < q


def wilson_lower(k, n, z=1.96):
    if n <= 0:
        return math.nan
    ph, z2 = k / n, z * z
    return (ph + z2 / (2 * n) - z * math.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n))) / (1 + z2 / n)


def h1_decision(p, est):
    if not (np.isfinite(p) and np.isfinite(est)):
        return "not evaluable"
    if p < H1_ALPHA and est > H1_MARGIN:
        return "D better"
    if p < H1_ALPHA and est < -H1_MARGIN:
        return "A better"
    return "inconclusive"


def ols_d_effect(df):
    """Coefficient of the D indicator in improvement ~ 1 + isD + slots_completed (A and D runs only)."""
    df = df[df.arm.isin(["A", "D"])].dropna(subset=["improvement", "slots_completed"])
    if len(df) < 4 or df.arm.nunique() < 2:
        return math.nan
    X = np.column_stack([np.ones(len(df)), (df.arm == "D").to_numpy(float), df.slots_completed.to_numpy(float)])
    return float(np.linalg.lstsq(X, df.improvement.to_numpy(float), rcond=None)[0][1])


# ---- per loop run features ------------------------------------------------------------------
def is_open(model):
    return bool(model) and not any(s in str(model).lower() for s in CLOSED)


def read_routing(x):
    """(table, source) from a routing dict or the path of a routing_table.json (relative paths: cwd, then REPO), else None."""
    if isinstance(x, dict) and x:
        return x, "inline routing"
    if isinstance(x, str) and x:
        for p in (Path(x), REPO / x):
            t = read_json(p)
            if isinstance(t, dict) and t:
                return t, str(p)
    return None


def find_routing(run, plan):
    """The frozen routing table behind the ledger-derived served fraction, as (table, source) or (None, None): the
    `routing` of a D loop config listed in plan.json (or plan.json's own `routing`), else <run>/routing_table_frozen.json,
    else <run>/../calibration/routing_table.json."""
    cands = [plan.get("routing")]
    for lp in plan.get("loops") or []:
        if isinstance(lp, dict) and lp.get("arm") == "D" and lp.get("config"):
            cands.append((read_json(lp["config"]) or {}).get("routing"))
    cands += [str(Path(run) / "routing_table_frozen.json"), str(Path(run).parent / "calibration" / "routing_table.json")]
    for c in cands:
        got = read_routing(c)
        if got:
            return got
    return None, None


def designated_model(table, role, edit_class):
    """The model the frozen table sends this role to (mirrors the engine's _route; None when the table has no entry)."""
    if role == "coder" and edit_class in (table.get("coder_by_class") or {}):
        return table["coder_by_class"][edit_class]
    keys = ("planner_refresh", "planner_cheap", "planner") if role == "planner_refresh" else (role,)
    return next((table[k] for k in keys if isinstance(table.get(k), str)), None)


def ledger_served(recs, table):
    """(designated, open_served) over the ledger model_calls: a call is designated when the frozen table sends its role
    (coder by the record's edit_class) to an open model, and open-served when its served_model is open. Unlike the
    state counters this includes calls that a failed pre-slot health check rerouted to the fallback."""
    des = srv = 0
    for r in recs:
        for c in r.get("model_calls") or []:
            role = str(c.get("role") or "").lower()
            if role in OPEN_ROLES and is_open(designated_model(table, role, r.get("edit_class"))):
                des += 1
                srv += is_open(c.get("served_model"))
    return des, srv


def lint_failed(r):
    lint = r.get("lint")
    if isinstance(lint, dict):
        return lint.get("ok") is False or bool(lint.get("hard"))
    kind = str(r.get("kind") or "").lower()
    return "lint" in kind and "fail" in kind


def ledger_features(recs, sigma):
    f = dict(accepts=0, replicates=0, audits=0, audit_rejects=0, audit_accepts=0, integrity_flags=0, lint_failures=0, bad_accepts=0,
             model_usd_plan=0.0, model_usd_nominal=0.0, model_calls=0)
    trip = 0
    for r in recs:
        kind = str(r.get("kind") or "").lower()
        flags = [str(x).lower() for x in (r.get("flags") or [])]
        trip += any("tripwire" in s for s in flags)
        f["integrity_flags"] += sum("tripwire" not in s for s in flags)
        f["replicates"] += "replic" in kind
        f["audits"] += "audit" in kind
        f["lint_failures"] += bool(lint_failed(r))
        decision = str(r.get("decision") or "").lower()
        f["audit_rejects"] += "audit" in kind and decision.startswith("reject")
        f["audit_accepts"] += "audit" in kind and decision.startswith("accept")
        if decision.startswith("accept"):
            f["accepts"] += 1
            before = num((r.get("incumbent_before") or {}).get("mean"))
            after = num((r.get("incumbent_after") or {}).get("mean"))
            cand = after if not math.isnan(after) else num(r.get("adjusted_val_bpb"))
            if before - cand < sigma:   # NaN compares False
                f["bad_accepts"] += 1
        for c in r.get("model_calls") or []:
            f["model_calls"] += 1
            f["model_usd_plan"] += num0(c.get("cost_usd_plan"))
            f["model_usd_nominal"] += num0(c.get("cost_usd_nominal"))
    f["tripwires"] = max(trip, f["audits"])
    return f


def confirm_summary(conf):
    """Per loop_id means over its usable confirmation runs, each paired with the same-wave anchor."""
    cols = ["n_confirm", "confirmed_official_mean", "confirmed_sealed_mean", "improvement",
            "confirmed_gain_official", "gap_shard"]
    empty = pd.DataFrame(columns=cols, index=pd.Index([], name="loop_id"))
    if conf.empty:
        return empty
    c = conf.reindex(columns=["wave", "loop_id", "kind", "official_val_bpb", "sealed_val_bpb", "status"]).copy()
    for col in ("official_val_bpb", "sealed_val_bpb"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c[(c.status.astype(str) == "ok") & c.official_val_bpb.notna() & c.sealed_val_bpb.notna()]
    anc, run = c[c.kind == "anchor"], c[c.kind == "confirm"].copy()
    if anc.empty or run.empty:
        return empty
    wave = anc.groupby("wave")[["official_val_bpb", "sealed_val_bpb"]].mean()
    glob = anc[["official_val_bpb", "sealed_val_bpb"]].mean()   # fallback when a wave lost its anchor
    a_off = run.wave.map(wave.official_val_bpb).fillna(glob.official_val_bpb)
    a_seal = run.wave.map(wave.sealed_val_bpb).fillna(glob.sealed_val_bpb)
    run["imp_sealed"] = a_seal - run.sealed_val_bpb
    run["imp_off"] = a_off - run.official_val_bpb
    run["gap_shard"] = (run.official_val_bpb - run.sealed_val_bpb) - (a_off - a_seal)
    return run.groupby("loop_id").agg(
        n_confirm=("sealed_val_bpb", "size"), confirmed_official_mean=("official_val_bpb", "mean"),
        confirmed_sealed_mean=("sealed_val_bpb", "mean"), improvement=("imp_sealed", "mean"),
        confirmed_gain_official=("imp_off", "mean"), gap_shard=("gap_shard", "mean"))


def build_table(loops, conf, anchor_mean, sigma, routing=None):
    cs = confirm_summary(conf).to_dict("index")
    rows = []
    for st, recs in loops:
        f = ledger_features(recs, sigma)
        lid = st["loop_id"]
        in_ledger = sum(1 for r in recs if r.get("job_id"))
        slots = num(st.get("slot_index"))
        slots = in_ledger if math.isnan(slots) else int(slots)
        claimed = num(st.get("claimed_best_official"))
        if math.isnan(claimed):
            claimed = anchor_mean   # nothing accepted: the loop claims the anchor, a gain of 0
        k = cs.get(lid, {})
        des, srv = num0(st.get("designated_open_calls")), num0(st.get("open_served_calls"))
        frac = srv / des if des > 0 else math.nan
        is_d = st.get("arm") == "D"
        ldes, lsrv = ledger_served(recs, routing) if (is_d and routing) else (des, srv)   # no table: state counters
        frac_l = lsrv / ldes if is_d and ldes > 0 else math.nan
        claimed_gain = anchor_mean - claimed
        gain_off = k.get("confirmed_gain_official", math.nan)
        rows.append(dict(
            arm=st.get("arm"), loop_id=lid, status=st.get("status"), slots_completed=slots, slots_ledger=in_ledger,
            valid=bool(slots >= MIN_VALID_SLOTS), accepts=f["accepts"], replicates=f["replicates"],
            audits=f["audits"], audit_rejects=f["audit_rejects"], audit_accepts=f["audit_accepts"],
            would_be_keeps=f["accepts"] + f["audit_rejects"],
            audit_reject_rate=f["audit_rejects"] / f["audits"] if f["audits"] else math.nan,
            integrity_flags=f["integrity_flags"], tripwires=f["tripwires"],
            lint_failures=f["lint_failures"], bad_accepts=f["bad_accepts"],
            claimed_best_official=claimed, n_confirm=k.get("n_confirm", 0),
            confirmed_official_mean=k.get("confirmed_official_mean", math.nan),
            confirmed_sealed_mean=k.get("confirmed_sealed_mean", math.nan),
            improvement=k.get("improvement", math.nan), claimed_gain=claimed_gain,
            confirmed_gain_official=gain_off, gap_sel=claimed_gain - gain_off,
            gap_shard=k.get("gap_shard", math.nan), model_usd_plan=f["model_usd_plan"],
            model_usd_nominal=f["model_usd_nominal"], model_calls=f["model_calls"], served_fraction=frac,
            served_fraction_ledger=frac_l, degraded=bool(is_d and ldes > 0 and frac_l < DEGRADED_BELOW)))
    cols = list(rows[0]) if rows else ["arm", "loop_id", "valid", "degraded"]
    tab = pd.DataFrame(rows, columns=cols)
    if rows:
        tab["integrity_plus_tripwires"] = tab.integrity_flags + tab.tripwires
        for kind in ("plan", "nominal"):
            tab["usd_per_accept_" + kind] = np.where(tab.accepts > 0, tab["model_usd_" + kind] / tab.accepts.clip(lower=1), np.nan)
        tab = tab.sort_values(["arm", "loop_id"], kind="stable").reset_index(drop=True)
    return tab


def grp(tab, arm, col, valid=True, clean=False):
    if tab.empty:
        return np.array([])
    t = tab[tab.arm == arm]
    if valid:
        t = t[t.valid]
    if clean:
        t = t[~t.degraded]
    return finite(t[col].to_numpy(float))


# ---- sentinels ------------------------------------------------------------------------------
def sentinel_check(sent, anchor):
    res = dict(n=0, sd_adjusted=math.nan, sd_raw=math.nan, ratio=math.nan, flagged=None, per_gpu=[], adj=None)
    if sent.empty:
        return res
    s = sent.reindex(columns=["ts", "job_id", "gpu", "official_val_bpb", "status", "slot_count_at_submit"]).copy()
    s["v"] = pd.to_numeric(s.official_val_bpb, errors="coerce")
    s = s[s.v.notna() & (s.status.astype(str) == "ok")].reset_index(drop=True)
    res["n"] = len(s)
    if s.empty:
        return res
    offs = {str(k): v for k, v in (anchor.get("gpu_offsets") or {}).items()}
    g = s.gpu.astype(str).str.extract(r"(\d+)\s*$")[0]
    s["off"] = pd.to_numeric(g.map(offs), errors="coerce").fillna(0.0)
    s["adj"] = s.v - s.off
    if len(s) > 1:
        res["sd_adjusted"], res["sd_raw"] = float(s.adj.std(ddof=1)), float(s.v.std(ddof=1))
        res["ratio"] = res["sd_adjusted"] / NOISE_RESID_SD
        res["flagged"] = bool(abs(res["ratio"] - 1) > SIGMA_TOL)
    grand = s.v.mean()
    for gpu, t in s.assign(gpu_key=g.fillna(s.gpu.astype(str))).groupby("gpu_key"):
        res["per_gpu"].append(dict(gpu=str(gpu), n=len(t), mean=float(t.v.mean()),
                                   sd=float(t.v.std(ddof=1)) if len(t) > 1 else math.nan,
                                   offset_observed=float(t.v.mean() - grand), offset_frozen=offs.get(str(gpu), math.nan)))
    res["adj"] = s
    return res


# ---- the hypotheses -------------------------------------------------------------------------
def qualify_rows(items):
    """items: iterable of (label dict, passes, n). Wilson 95% lower bound vs 0.6 (structural 0.75)."""
    rows = []
    for lab, k, n in items:
        bar = 0.75 if lab["edit_class"] == "structural" else 0.6
        lb = wilson_lower(k, n)
        rows.append(dict(**lab, n=int(n), passes=int(k), rate=k / n if n else math.nan, wilson_lower=lb, bar=bar,
                         qualifies=bool(lb >= bar)))
    return rows


def coder_pass_rows(loops):
    """Ledger proxy for H5: D-arm records whose first coder call was served by an open model; pass = lint ok
    and the GPU run finished with status ok."""
    cnt = {}
    for st, recs in loops:
        if st.get("arm") != "D":
            continue
        for r in recs:
            coder = next((c for c in r.get("model_calls") or [] if str(c.get("role") or "").lower().startswith("coder")), None)
            if coder and is_open(coder.get("served_model")):
                k = cnt.setdefault(str(r.get("edit_class") or "unknown"), [0, 0])
                k[1] += 1
                k[0] += (not lint_failed(r)) and str(r.get("status") or "").lower() == "ok"
    return qualify_rows(({"edit_class": ec}, k, n) for ec, (k, n) in sorted(cnt.items()))


def h5_met(rows):
    q = {r["edit_class"]: r["qualifies"] for r in rows}
    if not any(c in q for c in ("constant", "schedule", "structural")):
        return None
    return all(q[c] for c in ("constant", "schedule") if c in q) and not q.get("structural", False)


def routing_counts(loops):
    cnt = Counter()
    for st, recs in loops:
        if st.get("arm") == "D":
            for r in recs:
                for c in r.get("model_calls") or []:
                    cnt[(str(c.get("role") or "unknown"), str(c.get("served_model") or "unknown"))] += 1
    return cnt


SECONDARIES = [   # name, reference arm, treatment arm, column, drop degraded D
    ("gap_sel (A vs D)", "A", "D", "gap_sel", False),
    ("gap_shard (A vs D)", "A", "D", "gap_shard", False),
    ("accepts per run (A vs D)", "A", "D", "accepts", False),
    ("integrity + tripwire counts (A vs D)", "A", "D", "integrity_plus_tripwires", False),
    ("model dollars per run, nominal (A vs D)", "A", "D", "model_usd_nominal", False),
    ("D cost per accepted change, nominal (C vs D)", "C", "D", "usd_per_accept_nominal", True),
]


def analyze(tab, loops, sent, anchor, rng, calib=None):
    res = {}
    # H1
    a, d = grp(tab, "A", "improvement"), grp(tab, "D", "improvement")
    h1 = perm_test(a, d, rng, interval=True)
    h1.update(decision=h1_decision(h1["p"], h1["estimate"]), mean_A=float(a.mean()) if len(a) else math.nan,
              mean_D=float(d.mean()) if len(d) else math.nan, interval_method="test inversion, constant-shift model")
    res["h1"] = h1
    itt = perm_test(grp(tab, "A", "improvement", valid=False), grp(tab, "D", "improvement", valid=False), rng, interval=True)
    itt["ols_slots_adjusted_estimate"] = ols_d_effect(tab) if not tab.empty else math.nan
    nodeg = perm_test(a, grp(tab, "D", "improvement", clean=True), rng, interval=True)
    res["h1_sensitivity"] = {"itt_all_runs": itt, "valid_excluding_degraded_D": nodeg}
    # rung contrasts (later rung minus earlier rung)
    res["rungs"] = []
    for x, y in (("A", "B"), ("B", "C"), ("C", "D")):
        vx, vy = grp(tab, x, "improvement"), grp(tab, y, "improvement")
        est, lo, hi = boot_diff(vx, vy, rng)
        res["rungs"].append(dict(contrast=f"{y} minus {x}", n_ref=len(vx), n_treat=len(vy), estimate=est, lo=lo, hi=hi))
    # H2, H3
    res["h2"] = {arm: dict(zip(("mean", "lo", "hi"), boot_mean(grp(tab, arm, "gap_sel"), rng)), n=len(grp(tab, arm, "gap_sel"))) for arm in ARMS}
    res["h3"] = {arm: dict(zip(("mean", "lo", "hi"), boot_mean(grp(tab, arm, "bad_accepts"), rng)), n=len(grp(tab, arm, "bad_accepts"))) for arm in ARMS}
    m2 = {k: v["mean"] for k, v in res["h2"].items()}
    m3 = {k: v["mean"] for k, v in res["h3"].items()}
    ver2 = [m2[k] for k in "BCD" if np.isfinite(m2[k])]
    ver3 = [m3[k] for k in "BCD" if np.isfinite(m3[k])]
    res["h2_checks"] = dict(
        A_larger_than_each_verified=bool(np.isfinite(m2["A"]) and ver2 and all(m2["A"] > v for v in ver2)),
        A_in_0p010_to_0p013=bool(np.isfinite(m2["A"]) and 0.010 <= m2["A"] <= 0.013),
        verified_under_0p006=bool(ver2 and all(v < 0.006 for v in ver2)))
    res["h3_checks"] = dict(A_over_3=bool(np.isfinite(m3["A"]) and m3["A"] > 3),
                            verified_at_most_1=bool(ver3 and all(v <= 1 for v in ver3)))
    # H4 (D clean vs C)
    dc, cc = grp(tab, "D", "improvement", clean=True), grp(tab, "C", "improvement")
    est, lo, hi = boot_diff(cc, dc, rng)
    h4 = dict(n_D_clean=len(dc), n_C=len(cc), estimate_D_minus_C=est, lo=lo, hi=hi,
              noninferior=bool(np.isfinite(lo) and lo > -NONINF_MARGIN))
    for kind in ("plan", "nominal"):
        col = "usd_per_accept_" + kind
        md, mc = grp(tab, "D", col, clean=True), grp(tab, "C", col)
        h4[f"usd_per_accept_{kind}_D"] = float(md.mean()) if len(md) else math.nan
        h4[f"usd_per_accept_{kind}_C"] = float(mc.mean()) if len(mc) else math.nan
        h4[f"D_cheaper_{kind}"] = bool(len(md) and len(mc) and md.mean() < mc.mean())
    dt = tab[tab.arm == "D"] if not tab.empty else tab
    h4.update(n_D=len(dt), n_D_degraded=int(dt.degraded.sum()) if len(dt) else 0,
              served_fraction_mean=float(dt.served_fraction.mean()) if len(dt) else math.nan,
              served_fraction_ledger_mean=float(dt.served_fraction_ledger.mean()) if len(dt) else math.nan)
    res["h4"] = h4
    # H5
    res["h5_ledger"] = coder_pass_rows(loops)
    res["h5_ledger_met"] = h5_met(res["h5_ledger"])
    res["h5_calibration"] = []
    if calib:
        res["h5_calibration"] = qualify_rows(({"model": m, "edit_class": ec}, k, n)
                                             for m, d_ in sorted(calib.items()) for ec, (k, n) in sorted(d_.items()))
    # secondaries + BH over exactly these six
    sec = []
    for name, ref, trt, col, clean in SECONDARIES:
        t = perm_test(grp(tab, ref, col, clean=clean), grp(tab, trt, col, clean=clean), rng)
        sec.append(dict(name=name, contrast=f"{trt} minus {ref}", n_ref=t["n_x"], n_treat=t["n_y"],
                        estimate=t["estimate"], p=t["p"], method=t["method"]))
    q, rej = bh([s["p"] for s in sec])
    for s, qi, ri in zip(sec, q, rej):
        s.update(q=float(qi), significant=bool(ri))
    res["secondaries"] = sec
    # sentinels
    sc = sentinel_check(sent, anchor)
    res["sentinels"] = {k: v for k, v in sc.items() if k != "adj"}
    res["_sentinel_frame"] = sc["adj"]
    # power note
    na, nd = len(a), len(d)
    if na > 1 and nd > 1:
        sdw = math.sqrt(((na - 1) * a.var(ddof=1) + (nd - 1) * d.var(ddof=1)) / (na + nd - 2))
        res["power"] = dict(sd_within=sdw, mde_80=2.80 * sdw * math.sqrt(1 / na + 1 / nd))
    else:
        res["power"] = dict(sd_within=math.nan, mde_80=math.nan)
    return res


# ---- loading and the top-level pipeline -----------------------------------------------------
def load_run(run):
    run = Path(run)
    loops = []
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        st = read_json(d / "state.json")
        if isinstance(st, dict):
            st["loop_id"] = st.get("loop_id") or d.name
            loops.append((st, read_jsonl(d / "ledger.jsonl")))
    conf = pd.DataFrame(read_jsonl(run / "confirmation" / "results.jsonl"))
    sent = pd.DataFrame(read_jsonl(run / "sentinels.jsonl"))
    return loops, conf, sent, read_json(run / "anchor.json") or {}, read_json(run / "plan.json") or {}


def analyze_run(run, seed=1, calibration=None):
    loops, conf, sent, anchor, plan = load_run(run)
    sigma = num(plan.get("sigma_used"))
    sigma = SIGMA_USED if math.isnan(sigma) else sigma
    routing, routing_src = find_routing(run, plan)
    tab = build_table(loops, conf, num(anchor.get("mean", anchor.get("anchor_mean"))), sigma, routing)
    calib = read_json(calibration) if calibration else None
    res = analyze(tab, loops, sent, anchor, np.random.default_rng(seed), calib)
    res["meta"] = dict(run=str(run), seed=seed, sigma_used=sigma, n_loops=len(loops), n_confirm_rows=len(conf),
                       n_sentinels=len(sent), anchor_mean=num(anchor.get("mean", anchor.get("anchor_mean"))), routing_table=routing_src)
    return tab, res, loops, anchor


# ---- figures --------------------------------------------------------------------------------
# Arm colors are categorical slots 1 to 4 of the reference palette (adjacent pairs pass the CVD gates);
# marker shape is the secondary encoding for the yellow/orange pair on scatters.
PAL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
MARK = {"A": "o", "B": "s", "C": "^", "D": "D"}
MODEL_ORDER = ["sonnet", "haiku", "opus", "qwen35-9b", "gemma4-12b", "r1-qwen3-8b", "qwen35-4b"]   # fixed so colors follow the model
SERIES8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.edgecolor": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def _fig(nrows=1, ncols=1, size=(6.2, 4.2)):
    fig, axs = plt.subplots(nrows, ncols, figsize=size, squeeze=False, facecolor=SURF)
    for ax in axs.flat:
        ax.set_facecolor(SURF)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    return fig, axs


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def _none(ax, msg="no data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=INK2)


def fig_improvement(tab, path):
    fig, axs = _fig()
    ax = axs[0, 0]
    jit = np.random.default_rng(0)
    any_pt = False
    for i, arm in enumerate(ARMS):
        t = tab[(tab.arm == arm) & tab.improvement.notna()] if not tab.empty else tab
        if t.empty:
            continue
        any_pt = True
        y, ok = t.improvement.to_numpy(float), t.valid.to_numpy(bool)
        x = i + jit.uniform(-0.16, 0.16, len(t))
        ax.scatter(x[ok], y[ok], marker=MARK[arm], s=44, color=PAL[arm], edgecolor=INK, linewidth=0.5, zorder=3)
        ax.scatter(x[~ok], y[~ok], marker=MARK[arm], s=44, facecolor="none", edgecolor=PAL[arm], linewidth=1.5, zorder=3)
        if ok.any():
            ax.hlines(y[ok].mean(), i - 0.3, i + 0.3, color=INK, lw=2, zorder=4)
    if not any_pt:
        _none(ax)
    n = {a: int(tab[(tab.arm == a) & tab.improvement.notna()].shape[0]) if not tab.empty else 0 for a in ARMS}
    ax.set_xticks(range(4), [f"{a} (n={n[a]})" for a in ARMS])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("sealed improvement over same-wave anchor\n(val_bpb, higher is better)")
    ax.set_title("Improvement by arm (bar = mean of valid runs; hollow = under 12 slots)", fontsize=9)
    _save(fig, path)


def _arm_scatter(tab, xcol, ycol, path, xlabel, ylabel, title, diag=False):
    fig, axs = _fig()
    ax = axs[0, 0]
    t0 = tab[tab[xcol].notna() & tab[ycol].notna()] if not tab.empty else tab
    for arm in ARMS:
        t = t0[t0.arm == arm]
        if t.empty:
            continue
        ok = t.valid.to_numpy(bool)
        x, y = t[xcol].to_numpy(float), t[ycol].to_numpy(float)
        ax.scatter(x[ok], y[ok], marker=MARK[arm], s=46, color=PAL[arm], edgecolor=INK, linewidth=0.5, zorder=3, label=f"{arm} (n={len(t)})")
        ax.scatter(x[~ok], y[~ok], marker=MARK[arm], s=46, facecolor="none", edgecolor=PAL[arm], linewidth=1.5, zorder=3)
    if t0.empty:
        _none(ax)
    else:
        if diag:
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def fig_routing(counts, path):
    fig, axs = _fig(size=(6.8, 3.8))
    ax = axs[0, 0]
    if not counts:
        _none(ax, "no D model calls in the ledgers")
    else:
        def fam(m):   # fixed model -> color slot, so colors follow the model; unknown models fold into "other"
            return next((o for o in MODEL_ORDER if o in m.lower()), "other")
        agg = Counter()
        for (r, m), v in counts.items():
            agg[(r, fam(m))] += v
        roles = sorted({r for r, _ in agg})
        models = [m for m in MODEL_ORDER + ["other"] if any(k[1] == m for k in agg)]
        color = {m: SERIES8[i] for i, m in enumerate(MODEL_ORDER + ["other"])}
        for j, role in enumerate(roles):
            tot = sum(v for (r, _), v in agg.items() if r == role)
            left = 0.0
            for m in models:
                share = agg.get((role, m), 0) / tot
                ax.barh(j, share, left=left, color=color[m], edgecolor=SURF, linewidth=1.5)
                left += share
            ax.text(1.01, j, f"n={tot}", va="center", color=INK2)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color[m]) for m in models]
        ax.legend(handles, models, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
        ax.set_yticks(range(len(roles)), roles)
        ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of served calls")
    ax.set_title("Arm D routing: served model by role", fontsize=9)
    _save(fig, path)


def fig_sentinels(sframe, anchor, path):
    fig, axs = _fig()
    ax = axs[0, 0]
    mean = num(anchor.get("mean", anchor.get("anchor_mean")))
    if sframe is None or len(sframe) == 0:
        _none(ax, "no sentinels")
    else:
        x = pd.to_numeric(sframe.slot_count_at_submit, errors="coerce")
        x = x.to_numpy(float) if x.notna().all() else np.arange(len(sframe), dtype=float)
        mean = mean if np.isfinite(mean) else float(sframe.adj.mean())
        for k, a in ((2, 0.10), (1, 0.18)):
            ax.axhspan(mean - k * NOISE_RESID_SD, mean + k * NOISE_RESID_SD, color=PAL["A"], alpha=a, lw=0,
                       label=f"anchor mean +- {k} x 0.005857")
        ax.axhline(mean, color=PAL["A"], lw=1.2)
        ax.plot(x, sframe.adj.to_numpy(float), color=INK2, lw=0.8, zorder=2)
        ax.scatter(x, sframe.adj.to_numpy(float), s=26, color=INK, zorder=3, label="sentinel (GPU-offset adjusted)")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_xlabel("loop slots completed when the sentinel was submitted")
    ax.set_ylabel("official val_bpb of the unmodified baseline")
    ax.set_title("Sentinel runs against the noise band", fontsize=9)
    _save(fig, path)


def fig_timeline(tab, ledgers, path):
    fig, axs = _fig(2, 2, size=(9.2, 6.4))
    for ax, arm in zip(axs.flat, ARMS):
        t = tab[(tab.arm == arm)] if not tab.empty else tab
        pick = t[t.valid & t.improvement.notna()] if not t.empty else t
        pick = pick if not pick.empty else t
        if pick.empty:
            _none(ax, f"no arm {arm} runs")
            ax.set_title(f"Arm {arm}", fontsize=9)
            continue
        med = pick.improvement.median() if pick.improvement.notna().any() else 0.0
        row = pick.assign(dist=(pick.improvement.fillna(med) - med).abs()).sort_values(["dist", "loop_id"]).iloc[0]
        recs = [r for r in ledgers.get(row.loop_id, []) if np.isfinite(num(r.get("adjusted_val_bpb")))]
        xs = [num(r.get("slot")) if np.isfinite(num(r.get("slot"))) else i for i, r in enumerate(recs)]
        ys = [num(r["adjusted_val_bpb"]) for r in recs]
        acc = np.array([str(r.get("decision") or "").lower().startswith("accept") for r in recs], bool)
        bad = np.array([any("tripwire" not in str(f).lower() for f in (r.get("flags") or [])) for r in recs], bool)
        xs, ys = np.array(xs, float), np.array(ys, float)
        if len(recs):
            ax.scatter(xs[~acc], ys[~acc], s=22, color="#a8a7a1", zorder=2, label="not accepted")
            ax.scatter(xs[acc], ys[acc], s=70, marker="*", color=PAL[arm], edgecolor=INK, linewidth=0.5, zorder=4, label="accepted")
            ax.scatter(xs[bad], ys[bad], s=40, marker="x", color="#e34948", zorder=5, label="integrity flag")
            inc = [(x, num((r.get("incumbent_after") or {}).get("mean"))) for x, r in zip(xs, recs)]
            inc = [(x, v) for x, v in inc if np.isfinite(v)]
            if inc:
                ax.step(*zip(*inc), where="post", color=INK2, lw=1, zorder=1, label="incumbent mean")
            ax.legend(frameon=False, fontsize=7, loc="best")
        else:
            _none(ax, "no ledger records")
        ax.set_title(f"Arm {arm}: {row.loop_id}", fontsize=9)
        ax.set_xlabel("slot")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("adjusted val_bpb (lower is better)")
    _save(fig, path)


def make_figures(out, tab, res, loops, anchor):
    out = Path(out)
    ledgers = {st["loop_id"]: recs for st, recs in loops}
    fig_improvement(tab, out / "fig_improvement_by_arm.png")
    _arm_scatter(tab, "claimed_gain", "confirmed_gain_official", out / "fig_claimed_vs_confirmed.png",
                 "claimed gain over anchor mean, official (val_bpb)", "confirmed gain over same-wave anchor, official (val_bpb)",
                 "Claimed versus confirmed per loop run (below the dashed line = optimism)", diag=True)
    _arm_scatter(tab, "model_usd_nominal", "improvement", out / "fig_cost_vs_improvement.png",
                 "model dollars per loop run (nominal)", "sealed improvement (val_bpb)", "Model dollars versus improvement")
    fig_routing(routing_counts(loops), out / "fig_routing_shares_D.png")
    fig_sentinels(res.get("_sentinel_frame"), anchor, out / "fig_sentinels.png")
    fig_timeline(tab, ledgers, out / "fig_ledger_timeline.png")


FIGURES = ["fig_improvement_by_arm.png", "fig_claimed_vs_confirmed.png", "fig_cost_vs_improvement.png",
           "fig_routing_shares_D.png", "fig_sentinels.png", "fig_ledger_timeline.png"]


# ---- report ---------------------------------------------------------------------------------
def f(v, nd=5, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return "-inf" if v < 0 else "inf"
        return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return str(v)


def table_md(rows, cols, nd=5):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(r.get(c), nd) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def deviations_text(path):
    try:
        txt = Path(path).read_text()
    except OSError:
        return "(PREREG.md not found, deviations not copied)"
    m = re.search(r"^## Deviations[ \t]*\n(.*?)(?=^## |\Z)", txt, re.S | re.M)
    return m.group(1).strip() if m else "(no Deviations section found in PREREG.md)"


def h1_sentence(h1):
    if not np.isfinite(h1["p"]):
        return "H1 cannot be evaluated: it needs at least one valid loop run with confirmation data in both arm A and arm D."
    return (f"H1 verdict: {h1['decision'].upper()}. Sealed improvement D minus A = {f(h1['estimate'], 5, True)} val_bpb "
            f"(90% permutation interval [{f(h1['lo'], 5, True)}, {f(h1['hi'], 5, True)}] by test inversion), two-sided "
            f"{h1['method']} permutation p = {h1['p']:.4f} over {h1['n_x']} A and {h1['n_y']} D valid loop runs; "
            f"the pre-registered rule needs p < 0.05 and an estimate beyond 0.003 in absolute value.")


AUDIT_NOTE = (
    "Audit columns: once the cumulative tripwire trips (gain over the anchor above 0.03) every would-be accept is audited, in every arm "
    "including the control. audit_rejects counts audits that rejected, audit_accepts audits that confirmed, would_be_keeps = accepts + "
    "audit_rejects (what a keep-if-lower rule would have kept had audits not run; for arm A this is the pure Karpathy keep count) and "
    "audit_reject_rate = audit_rejects / audits (per-run values, averaged over valid runs that had audits). Audits in arm A are replication "
    "the control would not otherwise have had, so A's accepts are already filtered; audit_rejects in A estimate how many noise keeps the "
    "plain rule would have accepted.")


def served_note(src):
    if not src:
        return ("No routing table was found, so served_fraction_ledger falls back to the state counters (it equals served_fraction) "
                "and the degraded flag uses those; health-check reroutes are then not counted.")
    return (f"served_fraction_ledger is open-served over designated-open model calls in the ledger, using the frozen routing table "
            f"`{src}`; served_fraction comes from the state counters, which skip health-check reroutes. "
            f"The degraded flag (arm D, under {DEGRADED_BELOW}) uses served_fraction_ledger.")


def build_report(tab, res, prereg):
    h1, m = res["h1"], res["meta"]
    R = lambda df: df.to_dict("records") if len(df) else []   # noqa: E731
    L = ["# Analysis report", "",
         f"Run: `{m['run']}`. Seed {m['seed']}. {m['n_loops']} loop runs, {m['n_confirm_rows']} confirmation rows, "
         f"{m['n_sentinels']} sentinel rows. sigma_used {m['sigma_used']:.6f}. Anchor mean {f(m['anchor_mean'], 6)}.", "",
         "## H1 (confirmatory)", "", h1_sentence(h1), "",
         table_md([dict(arm="A", n=h1["n_x"], mean_improvement=h1["mean_A"]), dict(arm="D", n=h1["n_y"], mean_improvement=h1["mean_D"])],
                  ["arm", "n", "mean_improvement"]), "",
         "Sensitivity analyses (not the confirmatory test):", ""]
    s = res["h1_sensitivity"]
    L.append(table_md([
        dict(analysis="valid runs (confirmatory)", n_A=h1["n_x"], n_D=h1["n_y"], estimate=h1["estimate"], p=h1["p"], lo=h1["lo"], hi=h1["hi"]),
        dict(analysis="intention to treat, all runs", n_A=s["itt_all_runs"]["n_x"], n_D=s["itt_all_runs"]["n_y"], estimate=s["itt_all_runs"]["estimate"],
             p=s["itt_all_runs"]["p"], lo=s["itt_all_runs"]["lo"], hi=s["itt_all_runs"]["hi"]),
        dict(analysis="ITT, D coefficient adjusted for slots completed (OLS point estimate)", estimate=s["itt_all_runs"]["ols_slots_adjusted_estimate"]),
        dict(analysis="valid runs excluding degraded D", n_A=s["valid_excluding_degraded_D"]["n_x"], n_D=s["valid_excluding_degraded_D"]["n_y"],
             estimate=s["valid_excluding_degraded_D"]["estimate"], p=s["valid_excluding_degraded_D"]["p"],
             lo=s["valid_excluding_degraded_D"]["lo"], hi=s["valid_excluding_degraded_D"]["hi"]),
    ], ["analysis", "n_A", "n_D", "estimate", "p", "lo", "hi"]))
    L += ["", "## Rung contrasts (descriptive: estimate and 90% percentile bootstrap interval, no p-values)", "",
          table_md(res["rungs"], ["contrast", "n_ref", "n_treat", "estimate", "lo", "hi"]), "",
          "## Loop runs", "", "Process columns:", "",
          table_md(R(tab), ["arm", "loop_id", "slots_completed", "valid", "accepts", "replicates", "audits", "integrity_flags",
                            "tripwires", "lint_failures", "bad_accepts", "model_calls", "model_usd_plan", "model_usd_nominal",
                            "served_fraction", "served_fraction_ledger", "degraded"], 3), "", served_note(m["routing_table"]), "",
          "Outcome columns:", "",
          table_md(R(tab), ["arm", "loop_id", "n_confirm", "claimed_best_official", "confirmed_official_mean",
                            "confirmed_sealed_mean", "improvement", "gap_sel", "gap_shard"]), "",
          "## Arm summary (valid runs, mean)", ""]
    summ = []
    for arm in ARMS:
        t = tab[(tab.arm == arm)] if not tab.empty else tab
        v = t[t.valid] if not t.empty else t
        vm = lambda c: v[c].mean() if len(v) else math.nan   # noqa: E731
        summ.append(dict(arm=arm, runs=len(t), valid=len(v), improvement=v.improvement.mean() if len(v) else math.nan,
                         gap_sel=v.gap_sel.mean() if len(v) else math.nan, gap_shard=v.gap_shard.mean() if len(v) else math.nan,
                         accepts=v.accepts.mean() if len(v) else math.nan, bad_accepts=v.bad_accepts.mean() if len(v) else math.nan,
                         usd_plan=v.model_usd_plan.mean() if len(v) else math.nan, usd_nominal=v.model_usd_nominal.mean() if len(v) else math.nan,
                         audits=vm("audits"), audit_rejects=vm("audit_rejects"), audit_accepts=vm("audit_accepts"),
                         would_be_keeps=vm("would_be_keeps"), audit_reject_rate=vm("audit_reject_rate")))
    L += [table_md(summ, ["arm", "runs", "valid", "improvement", "gap_sel", "gap_shard", "accepts", "bad_accepts", "usd_plan", "usd_nominal",
                          "audits", "audit_rejects", "audit_accepts", "would_be_keeps", "audit_reject_rate"], 4), "", AUDIT_NOTE]
    L += ["", "## H2: selection gap (descriptive; positive = the loop's claim is optimistic)", "",
          table_md([dict(arm=a, **res["h2"][a]) for a in ARMS], ["arm", "n", "mean", "lo", "hi"]), "",
          "Predictions: " + ", ".join(f"{k} = {f(v)}" for k, v in res["h2_checks"].items()) + ".", "",
          "## H3: bad-accept proxy (accepted changes with single-step adjusted delta under sigma_used)", "",
          table_md([dict(arm=a, **res["h3"][a]) for a in ARMS], ["arm", "n", "mean", "lo", "hi"], 3), "",
          "Predictions: " + ", ".join(f"{k} = {f(v)}" for k, v in res["h3_checks"].items()) + ".", "",
          "## H4: routing (D without degraded runs versus C)", ""]
    h4 = res["h4"]
    L += [table_md([h4], ["n_D", "n_D_degraded", "served_fraction_mean", "served_fraction_ledger_mean", "n_D_clean", "n_C", "estimate_D_minus_C", "lo", "hi", "noninferior"], 4), "",
          table_md([h4], ["usd_per_accept_plan_D", "usd_per_accept_plan_C", "D_cheaper_plan", "usd_per_accept_nominal_D",
                          "usd_per_accept_nominal_C", "D_cheaper_nominal"], 4), "",
          "Noninferior means the lower end of the 90% bootstrap interval of D minus C improvement is above -0.01.", "",
          "## H5: open-model coder qualification", ""]
    if res["h5_ledger"]:
        L += ["Ledger proxy (arm D, coder calls served by open models; pass = lint ok and run status ok):", "",
              table_md(res["h5_ledger"], ["edit_class", "n", "passes", "rate", "wilson_lower", "bar", "qualifies"], 3), "",
              f"Prediction (constant and schedule qualify, structural does not): {f(res['h5_ledger_met'])}.", ""]
    else:
        L += ["No open-model coder calls in the D ledgers.", ""]
    if res["h5_calibration"]:
        L += ["Calibration table:", "", table_md(res["h5_calibration"], ["model", "edit_class", "n", "passes", "rate", "wilson_lower", "bar", "qualifies"], 3), ""]
    L += ["## Secondaries (permutation test of the difference of means, Benjamini-Hochberg q < 0.10 over these six)", "",
          table_md(res["secondaries"], ["name", "contrast", "n_ref", "n_treat", "estimate", "p", "method", "q", "significant"], 4), "",
          "## Live sigma check", ""]
    sn = res["sentinels"]
    if sn["flagged"] is None:
        L.append(f"Fewer than two usable sentinels ({sn['n']}); sigma check not possible.")
    else:
        L.append(f"Sentinel sd (GPU-offset adjusted) {f(sn['sd_adjusted'], 6)} against the noise-job residual sd {NOISE_RESID_SD} "
                 f"(ratio {f(sn['ratio'], 3)}, n = {sn['n']}; raw sd {f(sn['sd_raw'], 6)}). "
                 + ("FLAGGED: off by more than 25 percent, so the B, C, D decisions must be replayed offline with the sentinel sigma and both versions reported (replay is not implemented in this script)."
                    if sn["flagged"] else "Within 25 percent, no replay needed."))
    L += ["", "Per-GPU sentinel means (raw official val_bpb):", "",
          table_md(sn["per_gpu"], ["gpu", "n", "mean", "sd", "offset_observed", "offset_frozen"], 5), "", "## Figures", ""]
    L += [f"![{n[:-4]}]({n})" for n in FIGURES]
    L += ["", "## Deviations (copied from PREREG.md)", "", deviations_text(prereg), "", "## Power and limits", ""]
    pw = res["power"]
    L.append(f"With {h1['n_x']} A and {h1['n_y']} D valid runs and an observed within-arm sd of {f(pw['sd_within'], 4)}, the minimum "
             f"detectable D minus A difference at 80% power is about {f(pw['mde_80'], 4)} val_bpb (normal approximation; PREREG states "
             f"roughly 0.5 power at a true 0.008), so a non-significant H1 would not be evidence of no effect; B and C rest on about 3 runs each "
             f"and are descriptive only, model cost is not randomized, arm A is a stateless one-shot control and not Karpathy's persistent "
             f"session, and the harness tail defends against accidents and opportunism, not a deliberate adversary.")
    return "\n".join(L) + "\n"


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf") if math.isinf(o) else float(o)
    return o


def run_all(run, out, seed=1, prereg=None, calibration=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prereg = prereg or Path(__file__).resolve().parent.parent / "design" / "PREREG.md"
    tab, res, loops, anchor = analyze_run(run, seed, calibration)
    tab.to_csv(out / "loop_runs.csv", index=False)
    (out / "results.json").write_text(json.dumps(clean_json(res), indent=1))
    make_figures(out, tab, res, loops, anchor)
    # house rule: no em dashes in any output, whatever the inputs contained
    (out / "report.md").write_text(build_report(tab, res, prereg).replace(chr(0x2014), "-"))
    return tab, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run directory (contains the loop dirs, confirmation/, sentinels.jsonl, anchor.json)")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--seed", type=int, default=1, help="seed for Monte Carlo permutations and bootstraps")
    ap.add_argument("--prereg", default=None, help="PREREG.md to copy the Deviations section from")
    ap.add_argument("--calibration", default=None, help='optional JSON {"model": {"edit_class": [passes, n]}} for the H5 table')
    a = ap.parse_args(argv)
    tab, res = run_all(a.run, a.out, a.seed, a.prereg, a.calibration)
    print(h1_sentence(res["h1"]))
    print(f"wrote {a.out}/report.md ({len(tab)} loop runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
