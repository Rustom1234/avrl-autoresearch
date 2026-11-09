"""P3 analysis: does reasoning effort change Haiku's confidence as an escalation trigger, and is model-first better
than effort-first at equal dollars? Design and decision rules: runs/P3/PREREG.md.

Per effort level (default, low, medium, high) and per split (test is primary, burn is reported separately):
  - single-pass correctness against truth (ar3_features.judge with k=1): a clean control is correct iff no problem
    lines, a plant is correct iff a reported line is in accept_lines
  - accuracy (unweighted and with AR3 weights w), AUROC of (100 - judgment_confidence) for a wrong verdict,
    mean cost and output tokens per call
  - escalation frontier: the fraction q of items with the lowest confidence goes to the stored AR3 Sonnet pass 1
    (q = 0, 0.1, ..., 1), cost = Haiku cost + escalated Sonnet cost; items tied at the cutoff are escalated in
    equal fractions so the result is the exact expectation
  - H1: AUROC(high) - AUROC(low). H2: model-first (Haiku low + escalation) minus effort-first (Haiku high alone),
    accuracy at equal mean dollars by linear interpolation of the frontier
  - entry-clustered paired bootstrap (ar3_stats.cluster_bootstrap, 2000 draws, clusters = entry_key), 90 percent interval
  - primary statistics use the AR3 pi=0.25 weights w (the AR3 estimand); the unweighted versions are the sensitivity analysis

Only complete cases enter an analysis: items with a valid Haiku row at every analysed level and a valid AR3 Sonnet pass 1.

  python avrl/exp/p3_effort_analyze.py --splits test,burn --out avrl/runs/P3/analysis.json
  python avrl/exp/p3_effort_analyze.py --items day0 --splits burn --out avrl/runs/P3/day0_smoke_analysis.json --day0-gate
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ar3_features as F  # noqa: E402
import ar3_stats as S  # noqa: E402
import p3_effort_run as P  # noqa: E402

Q_GRID = tuple(round(0.1 * i, 1) for i in range(11))
H1_MARGIN = 0.05  # absolute AUROC difference of interest
CONF = 0.90
MF_LEVEL, EF_LEVEL = "low", "high"  # model-first escalates from Haiku low; effort-first is Haiku high alone
DAY0_MIN_RATIO, DAY0_MIN_VALID = 1.3, 0.95


# ---------------------------------------------------------------- loading

def correct_k1(item, lines):
    """Single-pass correctness: the pass's reported lines judged by ar3_features.judge (True/False, None if truth is not scorable)."""
    return F.judge(item["truth"], item["plant"], F.supported([set(lines)], 1))[1]


def load_level(run_dir, level):
    """{item_id: row}: the first valid row per item from calls_<level>.jsonl."""
    out = {}
    for r in P.read_rows(run_dir, level):
        if r.get("ok") and P.valid_output(r.get("structured_output")):
            out.setdefault(r["item_id"], r)
    return out


def call_stats(run_dir, level, ids):
    """Row-level bookkeeping for one level over the given item ids (all rows, valid or not)."""
    rows = [r for r in P.read_rows(run_dir, level) if r.get("item_id") in ids]
    ok = {r["item_id"] for r in rows if r.get("ok") and P.valid_output(r.get("structured_output"))}
    return {"rows": len(rows), "items_with_valid_row": len(ok), "items_expected": len(ids),
            "invalid_rows": sum(not (r.get("ok") and P.valid_output(r.get("structured_output"))) for r in rows),
            "fallback_rows": sum(bool(r.get("fallback")) for r in rows), "extra_attempts": sum(max(0, (r.get("attempts") or 1) - 1) for r in rows),
            "total_cost_usd": sum(r.get("cost_usd") or 0.0 for r in rows),
            "model_ids": sorted({m for r in rows for m in (r.get("model_ids") or [])}),
            "argv_efforts": sorted({str(r.get("argv_effort")) for r in rows}),
            "cli_versions": sorted({r.get("cli_version") or "" for r in rows})}


def load_sonnet_pass1(ar3_dir, ids):
    """{item_id: (parsed, cost)} from the stored AR3 Sonnet pass 1 (first valid record wins). Opens the test split only if frozen.json matches."""
    F.check_frozen(ar3_dir)
    out = {}
    for split in ("burn", "test"):
        p = Path(ar3_dir) / f"calls_{split}.jsonl"
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids and r.get("ok") and "sonnet" in str(r.get("model_alias", "")).lower() and int(r.get("pass_idx", 0)) == 1:
                parsed = F.parse_pass(r.get("structured_output"))
                if parsed and r["item_id"] not in out:
                    out[r["item_id"]] = (parsed, float(r.get("total_cost_usd") or 0.0))
    return out


def load_weights(ar3_dir):
    df = pd.read_parquet(Path(ar3_dir) / "items.parquet", columns=["item_id", "w"])
    return dict(zip(df.item_id, df.w))


def build_frame(items, rows_by_level, sonnet, weights, levels):
    """One row per complete-case item: truth, AR3 weight, Sonnet pass 1 correctness and cost, and per level
    cor_, conf_, cost_, tok_, lat_ columns."""
    recs = []
    for it in items:
        iid = it["item_id"]
        if iid not in sonnet or any(iid not in rows_by_level[lv] for lv in levels):
            continue
        rec = {"item_id": iid, "entry_key": it["entry_key"], "split": it["split"], "kind": it["kind"], "truth": it["truth"],
               "w": float(weights[iid]), "s_cor": float(correct_k1(it, sonnet[iid][0]["lines"])), "s_cost": sonnet[iid][1]}
        for lv in levels:
            r = rows_by_level[lv][iid]
            so = r["structured_output"]
            lines = F.parse_pass(so)["lines"]
            rec.update({f"cor_{lv}": float(correct_k1(it, lines)), f"conf_{lv}": float(so["judgment_confidence"]),
                        f"cost_{lv}": float(r["cost_usd"]), f"tok_{lv}": float((r.get("usage") or {}).get("output") or 0),
                        f"lat_{lv}": float(r.get("latency_s") or 0.0)})
        recs.append(rec)
    base = ["item_id", "entry_key", "split", "kind", "truth", "w", "s_cor", "s_cost"]
    return pd.DataFrame(recs, columns=base + [f"{c}_{lv}" for lv in levels for c in ("cor", "conf", "cost", "tok", "lat")])


# ---------------------------------------------------------------- statistics

def wmean(x, w=None):
    x = np.asarray(x, float)
    return float(np.average(x, weights=w)) if len(x) else float("nan")


def auroc_wrong(cor, conf, w=None):
    """AUROC of (100 - confidence) for a wrong verdict (positive class = wrong); ties count one half."""
    return S.weighted_auroc(1 - np.asarray(cor, float), 100 - np.asarray(conf, float), w)


def escalation_probs(conf, q):
    """Per-item escalation probability when the fraction q of items with the lowest confidence is escalated.
    Items below the cutoff value get 1, items at the cutoff share the remainder equally, the rest get 0."""
    conf = np.asarray(conf, float)
    n = len(conf)
    m = min(max(q, 0.0), 1.0) * n
    e = np.zeros(n)
    if m <= 1e-12:
        return e
    k = int(math.floor(m + 1e-9))
    if k >= n:
        return np.ones(n)
    cut = np.sort(conf)[k]
    below, at = conf < cut, conf == cut
    e[below] = 1.0
    e[at] = (m - below.sum()) / at.sum()
    return e


def frontier(h_cor, h_cost, conf, s_cor, s_cost, w=None, qs=Q_GRID):
    """Mean cost and accuracy at each escalation fraction q: {"q": [...], "cost": [...], "acc": [...]}."""
    h_cor, h_cost, s_cor, s_cost = (np.asarray(a, float) for a in (h_cor, h_cost, s_cor, s_cost))
    cost, acc = [], []
    for q in qs:
        e = escalation_probs(conf, q)
        cost.append(wmean(h_cost + e * s_cost, w))
        acc.append(wmean((1 - e) * h_cor + e * s_cor, w))
    return {"q": list(qs), "cost": cost, "acc": acc}


def acc_at_cost(costs, accs, target):
    """Accuracy at an exact mean cost by linear interpolation along a frontier; nan outside its cost range."""
    costs, accs = np.asarray(costs, float), np.asarray(accs, float)
    if not (costs[0] - 1e-15 <= target <= costs[-1] + 1e-15):
        return float("nan")
    return float(np.interp(target, costs, accs))


def _w(fr, weighted):
    return fr["w"].values if weighted else None


def level_frontier(fr, level, weighted=False):
    return frontier(fr[f"cor_{level}"], fr[f"cost_{level}"], fr[f"conf_{level}"], fr["s_cor"], fr["s_cost"], _w(fr, weighted))


def auroc_level(fr, level, weighted=False):
    return auroc_wrong(fr[f"cor_{level}"], fr[f"conf_{level}"], _w(fr, weighted))


def auroc_diff(fr, hi, lo, weighted=False):
    return auroc_level(fr, hi, weighted) - auroc_level(fr, lo, weighted)


def matched_cost_diff(fr, mf, ef, weighted=False):
    """Accuracy of model-first (Haiku at level mf plus escalation to Sonnet) minus effort-first (Haiku at level ef alone),
    with model-first's escalation fraction set so its mean cost equals effort-first's mean cost."""
    w = _w(fr, weighted)
    fm = level_frontier(fr, mf, weighted)
    return acc_at_cost(fm["cost"], fm["acc"], wmean(fr[f"cost_{ef}"], w)) - wmean(fr[f"cor_{ef}"], w)


def cluster_ci(fr, stat_fn, B, seed):
    """Entry-clustered bootstrap of stat_fn(frame), two-sided CONF interval. All statistics bootstrapped with the same
    seed see the same resampled entries, so contrasts are paired."""
    r = S.cluster_bootstrap(fr, "entry_key", stat_fn, B=B, seed=seed, conf=CONF)
    return {"est": r["est"], "lo": r["lo"], "hi": r["hi"], "n_clusters": r["n_clusters"], "draws_ok": r["B_ok"], "draws": B}


def h1_verdict(ci, margin=H1_MARGIN):
    lo, hi, est = ci["lo"], ci["hi"], ci["est"]
    if any(math.isnan(v) for v in (lo, hi, est)):
        return "undefined"
    if (lo > 0 or hi < 0) and abs(est) >= margin:
        return "high_better" if est > 0 else "high_worse"
    if lo > -margin and hi < margin:
        return "no_effect_within_margin"
    return "inconclusive"


def h2_verdict(ci):
    lo, hi = ci["lo"], ci["hi"]
    if math.isnan(lo) or math.isnan(hi):
        return "undefined"
    return "model_first_better" if lo > 0 else "effort_first_better" if hi < 0 else "no_difference"


# ---------------------------------------------------------------- per split

def level_table(fr, level):
    c, k, t = fr[f"cor_{level}"], fr[f"conf_{level}"], fr[f"tok_{level}"]
    return {"n": int(len(fr)), "n_wrong": int((c == 0).sum()), "accuracy": wmean(c), "accuracy_w": wmean(c, fr.w),
            "auroc": auroc_level(fr, level), "auroc_w": auroc_level(fr, level, True),
            "mean_conf": wmean(k), "mean_cost_usd": wmean(fr[f"cost_{level}"]), "mean_cost_usd_w": wmean(fr[f"cost_{level}"], fr.w),
            "mean_out_tokens": wmean(t), "median_out_tokens": float(np.median(t)) if len(t) else float("nan"),
            "mean_latency_s": wmean(fr[f"lat_{level}"])}


def analyze_split(fr, levels, B=2000, seed=20261106, mf=MF_LEVEL, ef=EF_LEVEL):
    """Everything for one split. Primary statistics use the AR3 pi=0.25 weights (the AR3 estimand); the same statistics
    without weights are the sensitivity analysis and sit under "unweighted"."""
    out = {"n_items": int(len(fr)), "n_entries": int(fr.entry_key.nunique()),
           "levels": {lv: level_table(fr, lv) for lv in levels},
           "sonnet_pass1": {"accuracy": wmean(fr.s_cor), "accuracy_w": wmean(fr.s_cor, fr.w), "mean_cost_usd": wmean(fr.s_cost)},
           "frontier": {lv: {"unweighted": level_frontier(fr, lv), "weighted": level_frontier(fr, lv, True)} for lv in levels}}
    if not len(fr):
        return out
    stats = {}
    for wt, sfx in ((True, "_w"), (False, "")):
        stats.update({f"auroc_{lv}{sfx}": (lambda f, lv=lv, wt=wt: auroc_level(f, lv, wt)) for lv in levels})
        if {mf, ef} <= set(levels):
            stats[f"auroc_{ef}_minus_{mf}{sfx}"] = lambda f, wt=wt: auroc_diff(f, ef, mf, wt)
            stats[f"matched_{mf}_to_{ef}{sfx}"] = lambda f, wt=wt: matched_cost_diff(f, mf, ef, wt)
        if {"default", ef} <= set(levels) and mf != "default":
            stats[f"matched_default_to_{ef}{sfx}"] = lambda f, wt=wt: matched_cost_diff(f, "default", ef, wt)
    out["ci90"] = {name: cluster_ci(fr, fn, B, seed) for name, fn in stats.items()}
    if {mf, ef} <= set(levels):
        def hyp(name, label, verdict):
            ci, ciu = out["ci90"][name + "_w"], out["ci90"][name]
            return {"contrast": label, "primary": "AR3 pi=0.25 weighted", **ci, "verdict": verdict(ci), "unweighted": {**ciu, "verdict": verdict(ciu)}}
        out["h1"] = hyp(f"auroc_{ef}_minus_{mf}", f"AUROC({ef}) - AUROC({mf})", h1_verdict)
        out["h2"] = hyp(f"matched_{mf}_to_{ef}", f"accuracy(model-first from {mf}) - accuracy({ef} alone) at equal mean dollars", h2_verdict)
        tok = float(fr[f"tok_{ef}"].median() / max(fr[f"tok_{mf}"].median(), 1e-9))
        out[f"token_ratio_median_{ef}_over_{mf}"] = tok
        out["kill_no_effort_spread_and_null"] = bool(tok < 1.5 and out["h1"]["verdict"] == "no_effect_within_margin")
    out["matched_cost_grid_point_estimates"] = {
        f"{a}_to_{b}": {"weighted": matched_cost_diff(fr, a, b, True), "unweighted": matched_cost_diff(fr, a, b)}
        for a in levels for b in levels if a != b}
    return out


# ---------------------------------------------------------------- day 0 gate

def day0_gate(run_dir, items, levels=P.LEVELS):
    """Day 0 gate numbers over the given items: token medians per level, validity, cost, latency, high/low ratio,
    and which level the default behaves like. PASS iff median output tokens(high) >= 1.3 x low and >= 95 percent valid."""
    ids = [it["item_id"] for it in items]
    per, rows = {}, {}
    for lv in levels:
        rows[lv] = load_level(run_dir, lv)
        got = [rows[lv][i] for i in ids if i in rows[lv]]
        toks = [(r.get("usage") or {}).get("output") or 0 for r in got]
        per[lv] = {"n_expected": len(ids), "n_valid": len(got), "validity": len(got) / len(ids) if ids else float("nan"),
                   "median_out_tokens": float(np.median(toks)) if toks else float("nan"), "mean_out_tokens": wmean(toks),
                   "mean_cost_usd": wmean([r["cost_usd"] for r in got]), "total_cost_usd": float(sum(r["cost_usd"] for r in got)),
                   "mean_latency_s": wmean([r.get("latency_s") or 0.0 for r in got]),
                   **{k: v for k, v in call_stats(run_dir, lv, set(ids)).items() if k in ("fallback_rows", "extra_attempts", "model_ids", "argv_efforts", "cli_versions")}}
    n_exp, n_val = sum(p["n_expected"] for p in per.values()), sum(p["n_valid"] for p in per.values())
    out = {"per_level": per, "validity_all": n_val / n_exp if n_exp else float("nan"),
           "total_cost_usd": float(sum(p["total_cost_usd"] for p in per.values()))}
    if "high" in per and "low" in per:
        out["median_ratio_high_over_low"] = per["high"]["median_out_tokens"] / max(per["low"]["median_out_tokens"], 1e-9)
        out["pass"] = bool(out["median_ratio_high_over_low"] >= DAY0_MIN_RATIO and out["validity_all"] >= DAY0_MIN_VALID)
    if "default" in per:
        d = per["default"]["median_out_tokens"]
        out["default_vs_levels"] = {lv: {"median_ratio_default_over_level": d / max(per[lv]["median_out_tokens"], 1e-9)}
                                    for lv in per if lv != "default"}
        for lv in out["default_vs_levels"]:
            both = [i for i in ids if i in rows["default"] and i in rows[lv]]
            lr = [math.log(max((rows["default"][i].get("usage") or {}).get("output") or 1, 1) / max((rows[lv][i].get("usage") or {}).get("output") or 1, 1)) for i in both]
            out["default_vs_levels"][lv]["mean_paired_log_ratio"] = wmean(lr)
        if out["default_vs_levels"]:
            near = min(out["default_vs_levels"], key=lambda lv: abs(math.log(max(out["default_vs_levels"][lv]["median_ratio_default_over_level"], 1e-9))))
            out["default_closest_level"] = near
            out["default_within_25pct_of_closest"] = bool(abs(math.log(max(out["default_vs_levels"][near]["median_ratio_default_over_level"], 1e-9))) <= math.log(1.25))
    return out


# ---------------------------------------------------------------- cli

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _fmt(v, d=3):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def print_split(name, res):
    print(f"\n== {name}: {res['n_items']} items, {res['n_entries']} entries ==")
    print("level     n  wrong   acc  acc_w  auroc auroc_w  $/call  out_tok(mean/med)  lat_s")
    for lv, t in res["levels"].items():
        print(f"{lv:8} {t['n']:3d} {t['n_wrong']:6d} {_fmt(t['accuracy'], 2):>5} {_fmt(t['accuracy_w'], 2):>6} {_fmt(t['auroc']):>6} {_fmt(t['auroc_w']):>7}"
              f"  {_fmt(t['mean_cost_usd'], 5)}  {t['mean_out_tokens']:8.0f}/{t['median_out_tokens']:<7.0f}  {_fmt(t['mean_latency_s'], 1)}")
    s = res["sonnet_pass1"]
    print(f"sonnet S1  acc {_fmt(s['accuracy'], 2)}  acc_w {_fmt(s['accuracy_w'], 2)}  $/call {_fmt(s['mean_cost_usd'], 5)}")
    for k in ("h1", "h2"):
        if k in res:
            h = res[k]
            print(f"{k.upper()} {h['contrast']}: {_fmt(h['est'])} [{_fmt(h['lo'])}, {_fmt(h['hi'])}] -> {h['verdict']}"
                  f" (unweighted {_fmt(h['unweighted']['est'])} [{_fmt(h['unweighted']['lo'])}, {_fmt(h['unweighted']['hi'])}] {h['unweighted']['verdict']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default=str(P.P3_DIR))
    ap.add_argument("--ar3-dir", default=str(P.AR3_DIR))
    ap.add_argument("--levels", default=",".join(P.LEVELS))
    ap.add_argument("--splits", default="test,burn")
    ap.add_argument("--items", choices=["all", "day0"], default="all")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", help="write the results JSON here")
    ap.add_argument("--day0-gate", action="store_true", help="also compute the Day 0 gate numbers (use with --items day0)")
    args = ap.parse_args(argv)
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    items, counts = P.select_items(args.ar3_dir)
    if args.items == "day0":
        items = P.day0_items(items)
    ids = {it["item_id"] for it in items}
    rows = {lv: load_level(args.run_dir, lv) for lv in levels}
    sonnet = load_sonnet_pass1(args.ar3_dir, ids)
    weights = load_weights(args.ar3_dir)
    result = {"meta": {"levels": levels, "items": args.items, "item_counts": counts, "B": args.B, "seed": args.seed, "ci": CONF,
                       "calls": {lv: call_stats(args.run_dir, lv, ids) for lv in levels}}, "splits": {}}
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        fr = build_frame([it for it in items if it["split"] == sp], rows, sonnet, weights, levels)
        result["splits"][sp] = analyze_split(fr, levels, args.B, args.seed)
        print_split(sp, result["splits"][sp])
    if args.day0_gate:
        result["day0_gate"] = day0_gate(args.run_dir, items, levels)
        print("\nDay 0 gate:", json.dumps(_clean(result["day0_gate"]), indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(_clean(result), indent=1) + "\n")
    return result


if __name__ == "__main__":
    main()
