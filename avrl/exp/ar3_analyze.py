"""AR3 analysis: burn-in and freeze, then the one-shot test analysis of the critic cascade.

Reads runs/AR3/items.parquet (written by ar3_features.py), uses ar3_stats.py for the statistics, and
follows exp/AR3_DESIGN.md sections 7 to 10.

  --step burn   burn split only. Diagnose (20 row shuffles), entry-level bootstrap of the burn-in,
                (t*, p*) at e=0.25, early gates, power appendix. Writes runs/AR3/frozen.json and
                frozen.sha256, or prints NOT TESTABLE and stops without freezing.
  --step test   refuses unless frozen.json matches frozen.sha256. Gates, reads R1 and R2, decision,
                secondary family (Holm), exploratory block, figures, result.json, table.md,
                cascade_config.json.
  --selftest    fabricates items.parquet with a perfectly informative and a random agreement signal,
                runs both steps end to end in a temp dir and asserts the section 8 replay unit tests.

Quality Q is the prevalence-weighted mean of per-item correctness (control: unflagged, plant: located).
Escalation rate e is a weight fraction. The escalated item takes sonnet_single_expect. Rank policies
escalate the lowest key first with ties split analytically (fractional escalation), so a curve at e is exact.
"""
import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar3_stats as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "AR3"
VENV_PY = "/homes/rdubash/active-audit/.venv/bin/python"

PI, M, E0, K = 0.25, 0.05, 0.25, 3
PLANTS = ["N-big", "N-mid", "N-stale", "V-body", "V-coh"]
FAMILY = {k: k[0] for k in PLANTS}
E_GRID = [0.10, 0.25, 0.40, 0.60]
FINE = np.round(np.linspace(0, 1, 201), 6)
SIGNALS = ["conf_agree", "conf_vote", "conf_verb", "conf_verb1"]
P7_FEATS = ["conf_agree", "conf_vote", "conf_verb", "mean_n_problems", "log_chars", "has_threshold_language", "flag_frac", "q_ok"]
G1_MIN, G0_RECALL, G0_FA, G0_POOL, G0_VALID = 0.06, 0.60, 0.35, 50, 0.95
BURN_RECALL, BURN_FA = 0.50, 0.40
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#666666", "black": "#000000"}

# parquet column names written by ar3_features.py (right) vs the names used in the design text (left)
ALIASES = {
    "weight": ("w",),
    "haiku_k1_correct": ("h1_correct",), "haiku_k3_correct": ("h3_correct",), "haiku_k5_correct": ("h5_correct",),
    "haiku_k3_flag": ("h3_flag",),
    "sonnet_pass_correct_1": ("s1_correct",), "sonnet_pass_correct_2": ("s2_correct",), "sonnet_pass_correct_3": ("s3_correct",),
    "haiku_cost_mean": ("haiku_cost_mean_usd",), "sonnet_cost_mean": ("sonnet_cost_mean_usd",),
}
CORE = ["conf_agree", "conf_vote", "conf_verb", "conf_verb1", "haiku_k1_correct", "haiku_k3_correct", "haiku_k3_flag", "sonnet_single_expect",
        "sonnet_majority_correct", "proxy_error_sonnet", "proxy_error_truth", "log_chars", "mean_n_problems", "has_threshold_language",
        "flag_frac", "weight"]
NEED = CORE + ["entry_key", "kind", "split", "q_ok", "haiku_k5_correct", "haiku_cost_mean", "sonnet_cost_mean"]

DEVIATIONS = []


def dev(msg):
    if msg not in DEVIATIONS:
        DEVIATIONS.append(msg)


# ---------------------------------------------------------------- io


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first_tok(p):
    """First whitespace-separated token of a text file, None if the file is missing or empty."""
    p = Path(p)
    tok = p.read_text().split() if p.exists() else []
    return tok[0] if tok else None


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


def wjson(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jclean(obj), indent=1))


def load_items(root, splits=None, drop_invalid=True):
    """Read items.parquet with the splits requested (predicate pushdown), canonical column names, no invalid or NaN-core rows."""
    if not (Path(root) / "items.parquet").exists():
        raise SystemExit(f"{Path(root) / 'items.parquet'} not found: run ar3_features.py first")
    flt = [("split", "in", list(splits))] if splits else None
    df = pd.read_parquet(Path(root) / "items.parquet", filters=flt)
    for canon, alts in ALIASES.items():
        if canon not in df.columns:
            for a in alts:
                if a in df.columns:
                    df[canon] = df[a]
                    break
    miss = [c for c in NEED if c not in df.columns]
    if miss:
        raise SystemExit(f"items.parquet is missing columns {miss}")
    n0 = len(df)
    if drop_invalid and "valid" in df.columns:
        df = df[df["valid"].astype(bool)]
    df = df.copy()
    df["has_threshold_language"] = df["has_threshold_language"].astype(float)
    df["haiku_cost_mean"] = df["haiku_cost_mean"].astype(float)
    df["sonnet_cost_mean"] = df["sonnet_cost_mean"].astype(float)
    if drop_invalid:
        df = df.dropna(subset=CORE)
    n_drop = n0 - len(df)
    n5 = int(df["haiku_k5_correct"].isna().sum())
    if n5:
        dev(f"{n5} items lack Haiku passes 4 and 5: haiku_k5_correct filled with haiku_k3_correct for them")
    df["haiku_k5_correct"] = df["haiku_k5_correct"].fillna(df["haiku_k3_correct"])
    df["conf_agree_10"] = df["conf_agree_10"].fillna(df["conf_agree"]) if "conf_agree_10" in df else df["conf_agree"]
    for c in ("haiku_cost_mean", "sonnet_cost_mean"):
        df[c] = df[c].fillna(df[c].mean())
    for c in SIGNALS + ["conf_agree_10"]:
        df[c] = df[c].astype(float).round(9)  # exact tie handling in ar3_stats.escalation_prob
    return df.reset_index(drop=True), n_drop


def arrays(df, p7=None):
    """Flat numpy view used by every statistic. Every value has length n so a bootstrap resample is one fancy index."""
    A = {
        "w": df["weight"].to_numpy(float), "qH1": df["haiku_k1_correct"].to_numpy(float), "qH3": df["haiku_k3_correct"].to_numpy(float),
        "qH5": df["haiku_k5_correct"].to_numpy(float), "qS": df["sonnet_single_expect"].to_numpy(float),
        "smaj": df["sonnet_majority_correct"].to_numpy(float), "flag": df["haiku_k3_flag"].to_numpy(float),
        "pe_s": df["proxy_error_sonnet"].to_numpy(float), "pe_t": df["proxy_error_truth"].to_numpy(float),
        "hc": df["haiku_cost_mean"].to_numpy(float), "sc": df["sonnet_cost_mean"].to_numpy(float),
        "agree": df["conf_agree"].to_numpy(float), "agree10": df["conf_agree_10"].to_numpy(float), "vote": df["conf_vote"].to_numpy(float),
        "verb": df["conf_verb"].to_numpy(float), "verb1": df["conf_verb1"].to_numpy(float),
        "len": -df["log_chars"].to_numpy(float), "npr": -df["mean_n_problems"].to_numpy(float),
    }
    if p7 is not None:
        A["p7"] = -p7_predict(p7, df)
    return A


def take(A, idx):
    return {k: v[idx] for k, v in A.items()}


def groups_of(df):
    codes, uniq = pd.factorize(df["entry_key"])  # first-appearance order, same as ar3_stats.cluster_bootstrap
    return codes, [np.flatnonzero(codes == j) for j in range(len(uniq))]


# ---------------------------------------------------------------- core replay maths


def esc_probs(key, w, e):
    """Per-item escalation probability so the escalated weight fraction is exactly e; lowest key first, tied groups share the boundary."""
    key = np.round(np.asarray(key, float), 9)
    order = np.argsort(key, kind="stable")
    ks, ws = key[order], w[order]
    new = np.r_[True, ks[1:] != ks[:-1]]
    starts = np.flatnonzero(new)
    gm = np.add.reduceat(ws, starts)
    before = np.cumsum(gm) - gm
    sg = np.clip((e * w.sum() - before) / np.where(gm > 0, gm, 1.0), 0.0, 1.0)
    s = np.empty(len(w))
    s[order] = sg[np.cumsum(new) - 1]
    return s


def wm(x, w):
    return float((w * x).sum() / w.sum())


def q_policy(base, qS, s, w):
    return wm(base + s * (qS - base), w)


def curve(base, qS, key, w, grid=FINE):
    return np.array([q_policy(base, qS, esc_probs(key, w, e), w) for e in grid])


def reach(grid, qcurve, target):
    ok = np.flatnonzero(qcurve >= target - 1e-12)
    return float(grid[ok[0]]) if len(ok) else None


def contributions(A, codes, E, cfg):
    """Per-entry additive contributions (each sums to the point estimate); entries are the units of the sign-flip tests."""
    w, W = A["w"], A["w"].sum()
    gain = A["qS"] - A["qH3"]
    s = S.escalation_prob(A["agree"], cfg["t"], cfg["p"])
    eh = (w * s).sum() / W
    s25 = esc_probs(A["agree"], w, E0)
    seq = esc_probs(A["agree"], w, cfg["e_eq"])
    per = {
        "G1": w * gain / W,
        "R1": w * (s - eh) * gain / W,
        "R2": w * (1 - s) * gain / W,
        "S3": w * (seq * gain + A["qH3"] - A["qH5"]) / W,
        "W": w / W,
    }
    if "p7" in A:
        per["S2"] = w * gain * (esc_probs(A["p7"], w, E0) - s25) / W
    return {k: np.bincount(codes, weights=v, minlength=E) for k, v in per.items()}


def compute(A, cfg):
    """Every bootstrapped quantity for one frame (full slice or a resample). Returns a flat dict of floats."""
    w, W = A["w"], A["w"].sum()
    qH3, qS = A["qH3"], A["qS"]
    gain = qS - qH3
    Q = lambda q: float((w * q).sum() / W)
    o = {"Q_S": Q(qS), "Q_H3": Q(qH3), "Q_H1": Q(A["qH1"]), "Q_H5": Q(A["qH5"]), "Q_Smaj": Q(A["smaj"])}
    o["G1"] = o["Q_S"] - o["Q_H3"]
    s = S.escalation_prob(A["agree"], cfg["t"], cfg["p"])
    eh = float((w * s).sum() / W)
    o["e_hat"] = eh
    o["Q_C"] = Q(qH3 + s * gain)
    o["Q_rand_eh"] = o["Q_H3"] + eh * o["G1"]
    o["Q_orc_eh"] = Q(qH3 + esc_probs(-gain, w, eh) * gain)
    o["R1"] = o["Q_C"] - o["Q_rand_eh"]
    o["R2"] = o["Q_S"] - o["Q_C"]
    den = o["Q_orc_eh"] - o["Q_rand_eh"]
    o["Gap"] = o["R1"] / den if den >= 0.01 else float("nan")
    for e in E_GRID:
        qc = Q(qH3 + esc_probs(A["agree"], w, e) * gain)
        qr = o["Q_H3"] + e * o["G1"]
        qo = Q(qH3 + esc_probs(-gain, w, e) * gain)
        o[f"Qc@{e}"], o[f"R1@{e}"], o[f"R2@{e}"] = qc, qc - qr, o["Q_S"] - qc
        o[f"Gap@{e}"] = (qc - qr) / (qo - qr) if qo - qr >= 0.01 else float("nan")
    if "p7" in A:
        o["S2"] = Q(gain * (esc_probs(A["p7"], w, E0) - esc_probs(A["agree"], w, E0)))
    o["S3"] = Q(qH3 + esc_probs(A["agree"], w, cfg["e_eq"]) * gain) - o["Q_H5"]
    for nm in ("agree", "vote", "verb", "verb1"):
        o[f"auc_s_{nm}"] = S.weighted_auroc(A["pe_s"], -A[nm], w)
        o[f"auc_t_{nm}"] = S.weighted_auroc(A["pe_t"], -A[nm], w)
    o["S1"] = o["auc_s_agree"] - o["auc_s_verb"]
    return o


def boot_all(A, groups, cfg, B, seed=0):
    """Cluster bootstrap of compute(): resample entries, keep item weights fixed (compute renormalizes). Same draws as
    ar3_stats.cluster_bootstrap for the same seed, so single statistics can be cross-checked against it."""
    rng = np.random.default_rng(seed)
    E = len(groups)
    rows = []
    for _ in range(B):
        draw = rng.integers(0, E, E)
        rows.append(compute(take(A, np.concatenate([groups[j] for j in draw])), cfg))
    return pd.DataFrame(rows)


def lo_hi(x, two=0.90):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    a = (1 - two) / 2 * 100
    return float(np.percentile(x, a)), float(np.percentile(x, 100 - a))


def lower95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, 5)) if len(x) else float("nan")


def upper95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, 95)) if len(x) else float("nan")


def replay_tests(A):
    """Section 8 unit tests; raises AssertionError if any fail. Returns what was checked."""
    w, qH3, qS = A["w"], A["qH3"], A["qS"]
    grid = FINE
    agree = curve(qH3, qS, A["agree"], w, grid)
    rand = curve(qH3, qS, np.zeros(len(w)), w, grid)
    gain = qS - qH3
    orc = curve(qH3, qS, -gain, w, grid)
    Q_H3, Q_S = wm(qH3, w), wm(qS, w)
    chk = {}
    chk["e0_equals_P0b"] = abs(agree[0] - Q_H3) < 1e-12
    chk["e1_equals_P1"] = abs(agree[-1] - Q_S) < 1e-12
    chk["random_is_linear"] = float(np.max(np.abs(rand - (Q_H3 + grid * (Q_S - Q_H3))))) < 1e-9
    i5 = int(np.argmin(np.abs(grid - 0.5)))
    chk["random_e05_is_midpoint"] = abs(rand[i5] - (Q_H3 + Q_S) / 2) < 1e-9
    others = [A[k] for k in ("agree", "vote", "verb", "verb1", "len", "npr") + (("p7",) if "p7" in A else ())]
    chk["oracle_maximal_everywhere"] = all(np.all(orc >= curve(qH3, qS, k, w, grid) - 1e-12) for k in others) and bool(np.all(orc >= rand - 1e-12))
    bad = [k for k, v in chk.items() if not v]
    assert not bad, f"replay unit tests failed: {bad}"
    return chk


# ---------------------------------------------------------------- burn-in helpers


def b_star_analytic(conf, err, w, rel=0.25):
    """Weighted analogue of diagnose._recommend_budget: smallest b on the 0.05 grid whose admitted error is <= rel * base error."""
    target = rel * wm(err, w)
    for i in range(1, 21):
        b = i / 20
        if wm((1 - esc_probs(conf, w, b)) * err, w) <= target + 1e-12:
            return b
    return 1.0


def replicate(df, cap=20000):
    w = df["weight"].to_numpy(float)
    base = w[w > 0].min()
    scale = 1.0
    while True:
        r = np.rint(w / base * scale).astype(int)
        if r.sum() <= cap:
            return df.loc[np.repeat(df.index.to_numpy(), r)].copy(), r
        scale *= cap / r.sum() * 0.98


def run_diagnose(frame, conf_cols, err_col, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    cols = list(conf_cols) + [err_col]
    sub = frame[cols]
    assert not sub.isna().any().any(), "NaN in the diagnose frame"
    pq = outdir / "frame.parquet"
    sub.to_parquet(pq, index=False)
    cmd = [VENV_PY, "-m", "avrl_router.diagnose", "--groups", str(pq), "--confidence", ",".join(conf_cols), "--error", err_col,
           "--budgets", "0.1,0.25,0.5", "--target_corruption", "rel:0.25", "--n_boot", "0", "--seed", "0", "--out", str(outdir)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"diagnose failed:\n{r.stderr[-2000:]}")
    pq.unlink()
    return json.loads((outdir / "diagnosis.json").read_text())


def diag_summary(d):
    out = {}
    for col, c in d["confidence_columns"].items():
        rb = c["recommended_budget"]
        out[col] = {"auc": c["auc"]["point"], "recommended_budget": 1.0 if rb is None or (isinstance(rb, float) and math.isnan(rb)) else rb,
                    "regime": c["regime"], "budgets": {b: v["threshold"]["point"] for b, v in c["budgets"].items()},
                    "random": {b: v["random"]["point"] for b, v in c["budgets"].items()}, "base_error": c["base_error"]}
    return out


def p7_fit(df):
    from sklearn.linear_model import LogisticRegression
    X = df[P7_FEATS].astype(float)
    med = X.median().fillna(0.0)
    X = X.fillna(med).to_numpy()
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    y = df["proxy_error_sonnet"].to_numpy(float)
    if len(np.unique(y)) < 2:
        return None
    w = df["weight"].to_numpy(float)
    m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y, sample_weight=w / w.mean())
    return {"features": P7_FEATS, "median": med.tolist(), "mean": mu.tolist(), "scale": sd.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}


def p7_predict(p7, df):
    X = df[p7["features"]].astype(float).fillna(pd.Series(p7["median"], index=p7["features"])).to_numpy()
    z = ((X - np.array(p7["mean"])) / np.array(p7["scale"])) @ np.array(p7["coef"]) + p7["intercept"]
    return 1 / (1 + np.exp(-z))


def critic_hashes():
    try:
        import ar3_run_critic as rc
        return {"system_sha256": rc.SYSTEM_SHA, "schema_sha256": rc.SCHEMA_SHA}
    except Exception:
        p = RUN / "critic_hashes.json"
        return json.loads(p.read_text()) if p.exists() else {"system_sha256": None, "schema_sha256": None}


def models_seen(root, files):
    seen = {}
    for f in files:
        p = Path(root) / f
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ids = r.get("model_ids") or []
            for m in (ids.keys() if isinstance(ids, dict) else ids):
                seen.setdefault(str(r.get("model_alias")), set()).add(m)
    return {k: sorted(v) for k, v in seen.items()}


def frame_digest(df):
    # weight is deliberately not hashed: relabeling a test control as natural after the freeze changes N_controls and so every control weight, burn ones included
    cols = ["entry_key", "kind", "conf_agree", "conf_vote", "conf_verb", "conf_verb1", "proxy_error_sonnet"]
    x = df[cols].round(9).sort_values(cols)
    return hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()


def macro_recall_fa(df):
    pl = df[df["kind"].isin(PLANTS)]
    per_kind = {k: float(g["sonnet_single_expect"].mean()) for k, g in pl.groupby("kind")}
    ctl = df[df["kind"] == "control"]
    return (float(np.mean(list(per_kind.values()))) if per_kind else float("nan"),
            float(1 - ctl["sonnet_single_expect"].mean()) if len(ctl) else float("nan"), per_kind)


def pool_unit_weights(df, pi):
    """Kind -> per-item weight and implied full-pool item count N_kind = (pi/5)/w (plants) or (1-pi)/w (controls)."""
    out = {}
    for k, g in df.groupby("kind"):
        w = float(g["weight"].median())
        nom = (1 - pi) if k == "control" else pi / 5
        out[k] = {"w_per_item": w, "n_pool": int(round(nom / w)) if w > 0 else None, "n_slice": int(len(g))}
    return out


# ---------------------------------------------------------------- step: burn


def step_burn(root, n_shuffle=20, n_boot_burn=1000, force=False):
    root = Path(root)
    if (root / "frozen.json").exists() and not force:
        raise SystemExit("frozen.json already exists; refusing to overwrite a freeze (use --force only to redo before any test look)")
    t0 = time.time()
    df, ndrop = load_items(root, ["burn"])
    if not len(df):
        raise SystemExit(f"no usable burn rows in {root / 'items.parquet'} ({ndrop} dropped as invalid or incomplete); nothing frozen")
    print(f"burn: {len(df)} items, {df['entry_key'].nunique()} entries ({ndrop} rows dropped as invalid or incomplete)", flush=True)
    A = arrays(df)
    codes, groups = groups_of(df)
    w = A["w"]
    print("replay unit tests (burn):", replay_tests(A), flush=True)

    # 5. early gates on burn-in (before any test row is opened)
    rec, fa, per_kind = macro_recall_fa(df)
    g0 = {"sonnet_macro_located_recall": rec, "sonnet_per_kind_recall": per_kind, "sonnet_fa": fa, "recall_min": BURN_RECALL, "fa_max": BURN_FA,
          "pass": bool(rec >= BURN_RECALL and fa <= BURN_FA)}
    print(f"G0-burn: Sonnet macro located recall {rec:.3f} (>= {BURN_RECALL}), FA {fa:.3f} (<= {BURN_FA}) -> {'pass' if g0['pass'] else 'FAIL'}", flush=True)
    if not g0["pass"]:
        print("NOT TESTABLE: early gate G0-burn failed; test rows stay closed, nothing frozen")
        wjson(root / "burn_report.json", {"status": "NOT TESTABLE", "g0_burn": g0})
        return 3

    # 1 and 2. replicated frame, diagnose over caller-side shuffles
    rep, reps = replicate(df)
    print(f"replicated burn frame: {len(rep)} rows (weight ratio max/min {w.max() / w[w > 0].min():.1f})", flush=True)
    diag_dir = root / "diag"
    sig_runs = []
    for sd in range(n_shuffle):
        fr = rep.iloc[np.random.default_rng(1000 + sd).permutation(len(rep))].reset_index(drop=True)
        sig_runs.append(diag_summary(run_diagnose(fr, SIGNALS, "proxy_error_sonnet", diag_dir / f"sonnet_s{sd:02d}")))
    fr = rep.iloc[np.random.default_rng(1000).permutation(len(rep))].reset_index(drop=True)
    truth_run = diag_summary(run_diagnose(fr, SIGNALS, "proxy_error_truth", diag_dir / "truth_s00"))
    b_runs = [r["conf_agree"]["recommended_budget"] for r in sig_runs]
    b_star = float(np.median(b_runs))
    auc_s = {c: sig_runs[0][c]["auc"] for c in SIGNALS}
    auc_t = {c: truth_run[c]["auc"] for c in SIGNALS}
    auc_w = {c: S.weighted_auroc(A["pe_s"], -A[c.replace("conf_", "")], w) for c in SIGNALS}
    print(f"diagnose conf_agree: b* median {b_star:.2f} over {n_shuffle} shuffles (min {min(b_runs):.2f}, max {max(b_runs):.2f}), regime {sig_runs[0]['conf_agree']['regime']}", flush=True)
    for c in SIGNALS:
        print(f"  AUROC {c:11s} vs sonnet {auc_s[c]:.3f} (weighted check {auc_w[c]:.3f})  vs truth {auc_t[c]:.3f}", flush=True)

    # 4. (t*, p*)
    t_s, p_s = S.weighted_quantile(A["agree"], w, E0)
    t_vote, p_vote = S.weighted_quantile(A["vote"], w, E0)
    t_b, p_b = S.weighted_quantile(A["agree"], w, b_star)
    print(f"t* = {t_s:.4f}, p* = {p_s:.4f} (e={E0}); conf_vote t={t_vote:.4f} p={p_vote:.4f}; at b*={b_star:.2f} t={t_b:.4f} p={p_b:.4f}", flush=True)

    # 3. entry-level bootstrap of the burn-in (weighted, analytic ties)
    rng = np.random.default_rng(0)
    bt = []
    for _ in range(n_boot_burn):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        Ab = take(A, idx)
        t, p = S.weighted_quantile(Ab["agree"], Ab["w"], E0)
        bt.append({"b": b_star_analytic(Ab["agree"], Ab["pe_s"], Ab["w"]), "t": t, "p": p, "dis": wm(Ab["pe_s"], Ab["w"])})
    bt = pd.DataFrame(bt)
    q = lambda x: [float(np.percentile(x, 5)), float(np.percentile(x, 95))]
    b_int, t_int, d_int = q(bt["b"]), q(bt["t"]), q(bt["dis"])
    b_an = b_star_analytic(A["agree"], A["pe_s"], w)
    print(f"burn bootstrap ({n_boot_burn}): b* 5-95% {b_int}, t* {t_int}, disagreement rate {wm(A['pe_s'], w):.3f} 5-95% {d_int} (analytic b* {b_an:.2f})", flush=True)
    bt.to_csv(root / "burn_bootstrap.csv", index=False)

    # P7 learned scorer, trained on burn only
    p7 = p7_fit(df)
    if p7 is None:
        dev("P7 not fit: proxy_error_sonnet has one class on the burn split")

    # power appendix: per-entry sd of (Q_S - Q_C) on burn at the frozen policy
    s = S.escalation_prob(A["agree"], t_s, p_s)
    u = np.bincount(codes, weights=w * (1 - s) * (A["qS"] - A["qH3"]) / w.sum(), minlength=len(groups)) * len(groups)
    sd_u = float(u.std(ddof=1))
    try:
        n_test = len({k for k, v in json.loads((root / "split.json").read_text()).items() if v == "test"})
    except FileNotFoundError:
        n_test = None
    power = {"sd_per_entry_QS_minus_QC_burn": sd_u, "n_test_entries": n_test, "mde_at_test_n": float(S.mde(sd_u, n_test)) if n_test else None,
             "burn_QS_minus_QC_in_sample": float(u.mean())}
    print("power appendix:", power, flush=True)

    kinds = pool_unit_weights(df, PI)
    pools = first_tok(root / "items.sha256")
    frozen = {
        "k": K, "signal": "conf_agree", "e_primary": E0, "t_star": t_s, "p_star": p_s, "b_star": b_star, "b_star_interval": b_int, "b_star_analytic": b_an,
        "t_star_interval": t_int, "burn_disagreement_rate": wm(A["pe_s"], w), "burn_disagreement_interval": d_int,
        "secondary": {"conf_vote_e25": {"t": t_vote, "p": p_vote}, "conf_agree_at_bstar": {"t": t_b, "p": p_b}},
        "burn_auroc_vs_sonnet": auc_s, "burn_auroc_vs_truth": auc_t, "burn_auroc_weighted_check": auc_w,
        "diagnose_b_star_runs": b_runs, "diagnose_regime": sig_runs[0]["conf_agree"]["regime"], "diagnose_budget_table": sig_runs[0]["conf_agree"],
        "M": M, "pi": PI, "weights_by_kind": kinds, "g0_burn": g0, "power_appendix": power,
        "burn_sonnet_error_rate_unweighted": float(1 - df["sonnet_single_expect"].mean()),
        "burn_sonnet_error_rate_weighted": float(1 - wm(A["qS"], w)),
        "items_sha256": pools, "split_sha256": sha256_file(root / "split.json") if (root / "split.json").exists() else None,
        "burn_frame_sha256": frame_digest(df), "n_burn_entries": len(groups), "n_burn_items": len(df),
        **critic_hashes(), "model_ids_seen": models_seen(root, ["calls_burn.jsonl"]), "p7": p7,
        "cost_per_call_burn": {"haiku": float(df["haiku_cost_mean"].mean()), "sonnet": float(df["sonnet_cost_mean"].mean())},
    }
    wjson(root / "frozen.json", frozen)
    (root / "frozen.sha256").write_text(f"{sha256_file(root / 'frozen.json')}  frozen.json\n")
    burn_figs(root, bt, b_star, t_s, b_int, t_int, d_int)
    print(f"froze {root / 'frozen.json'} in {time.time() - t0:.0f}s")
    return 0


def burn_figs(root, bt, b_star, t_s, b_int, t_int, d_int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = root / "fig"
    fd.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, col, ref, itv, ttl in zip(ax, ["b", "t", "dis"], [b_star, t_s, bt["dis"].mean()], [b_int, t_int, d_int],
                                       ["b* (analytic, burn bootstrap)", "t* at e=0.25", "Haiku vs Sonnet disagreement"]):
        a.hist(bt[col], bins=20, color=OI["sky"], edgecolor="white")
        a.axvline(ref, color=OI["black"], lw=1.5)
        a.axvspan(itv[0], itv[1], color=OI["grey"], alpha=0.15)
        a.set_title(ttl, fontsize=9)
    fig.tight_layout()
    fig.savefig(fd / "burn_bootstrap.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- step: test


def check_frozen(root):
    fj, fs = Path(root) / "frozen.json", Path(root) / "frozen.sha256"
    if not (fj.exists() and fs.exists()):
        raise SystemExit("refusing to run the test step: frozen.json or frozen.sha256 is missing")
    tok = fs.read_text().split()
    want, got = (tok[0].lower() if tok else ""), sha256_file(fj)
    if want != got:
        raise SystemExit(f"refusing to run the test step: frozen.json sha256 {got[:12]} != frozen.sha256 {want[:12]}")
    fz = json.loads(fj.read_text())
    root = Path(root)
    if fz.get("items_sha256") and first_tok(root / "items.sha256") != fz["items_sha256"]:
        raise SystemExit("refusing to run the test step: items.sha256 changed after the freeze")
    if fz.get("split_sha256") and sha256_file(root / "split.json") != fz["split_sha256"]:
        raise SystemExit("refusing to run the test step: split.json changed after the freeze")
    return fz


def merge_weights(df, pi, min_items=8):
    """Kinds with fewer than min_items test items are merged into their family (N-* or V-*): the family acts as one stratum,
    each of its items weighing |F| * (pi/5) / N_F with N_F the full-pool count recovered from the stored weights. Returns (w, merged families)."""
    t = df[df["split"] == "test"]
    cnt = t[t["kind"].isin(PLANTS)]["kind"].value_counts()
    fams = sorted({FAMILY[k] for k in PLANTS if cnt.get(k, 0) < min_items})
    w = df["weight"].to_numpy(float).copy()
    if not fams:
        return w, []
    pool = df[df["split"].isin(["burn", "test"]) & df["kind"].isin(PLANTS)]
    for f in fams:
        kinds = [k for k in PLANTS if FAMILY[k] == f]
        n_k = {k: (pi / 5) / float(pool[pool["kind"] == k]["weight"].median()) for k in kinds}
        wf = len(kinds) * (pi / 5) / sum(n_k.values())
        w[df["kind"].isin(kinds).to_numpy()] = wf
    return w, fams


def prow(sd, n, delta):
    """ar3_stats.power_row with the degenerate cases (zero sd or zero distance to the boundary) made explicit: q = inf or 0."""
    if sd > 0 and delta > 0:
        return S.power_row(sd, n, delta)
    return {"sd": float(sd), "n": int(n), "mde": float(S.mde(sd, n)), "delta": float(delta), "n_star": 0.0 if sd <= 0 else float("inf"), "q": float("inf") if sd <= 0 else 0.0}


def cost_ratio(k, e, hc, sc):
    return (k * hc + e * sc) / sc


def policy_defs():
    return {"P3_agree": ("qH3", "agree", 3), "P3b_vote": ("qH3", "vote", 3), "P4a_verb1": ("qH1", "verb1", 1), "P4b_verb": ("qH3", "verb", 3),
            "P7_learned": ("qH3", "p7", 3), "P8_longest": ("qH3", "len", 3), "P8_most_problems": ("qH3", "npr", 3), "P6_oracle": ("qH3", "oracle", 3),
            "P2_random": ("qH3", "zero", 3)}


def policy_curves(A):
    out = {}
    gain = A["qS"] - A["qH3"]
    for name, (base, key, k) in policy_defs().items():
        if key == "p7" and "p7" not in A:
            continue
        kv = {"oracle": -gain, "zero": np.zeros(len(gain))}.get(key, A.get(key))
        out[name] = {"k": k, "q": curve(A[base], A["qS"], kv, A["w"])}
    return out


def policy_table(A, curves, hc, sc, Q_S):
    rand3, orc3 = curves["P2_random"]["q"], curves["P6_oracle"]["q"]
    # a k=1 policy (P4a) escalates from the one-pass base, so its random and oracle references are built on that base, not on the k=3 base
    rand1 = curve(A["qH1"], A["qS"], np.zeros(len(A["w"])), A["w"])
    orc1 = curve(A["qH1"], A["qS"], -(A["qS"] - A["qH1"]), A["w"])
    T = {}
    for name, c in curves.items():
        q = c["q"]
        rand, orc = (rand1, orc1) if c["k"] == 1 else (rand3, orc3)
        den = orc - rand
        gap = np.where(den >= 0.01, (q - rand) / np.where(den >= 0.01, den, 1), np.nan)
        T[name] = {"Q_at_grid": {str(e): float(q[np.argmin(np.abs(FINE - e))]) for e in E_GRID},
                   "gap_closed_at_grid": {str(e): float(gap[np.argmin(np.abs(FINE - e))]) for e in E_GRID},
                   "e_for_98pct_Q_S": reach(FINE, q, 0.98 * Q_S), "e_for_99pct_Q_S": reach(FINE, q, 0.99 * Q_S),
                   "cost_ratio_at_grid": {str(e): cost_ratio(c["k"], e, hc, sc) for e in E_GRID}}
    return T


def pareto(points):
    pts = sorted(points)
    out, best = [], -1
    for c, q, n in pts:
        if q > best + 1e-12:
            out.append((c, q, n))
            best = q
    return out


def verdict_ge(lo_b, hi_b, thr):  # claim: effect > thr
    return "supported" if lo_b > thr else ("not supported" if hi_b <= thr else "unresolved")


def verdict_lt(lo_b, hi_b, thr):  # claim: effect < thr
    return "supported" if hi_b < thr else ("not supported" if lo_b >= thr else "unresolved")


def step_test(root, B=10000, n_sign=9999, n_splits=1000, B_sens=2000):
    root = Path(root)
    fz = check_frozen(root)
    t0 = time.time()
    df_all, _ = load_items(root, None, drop_invalid=True)
    pool = df_all[df_all["split"].isin(["burn", "test"])].reset_index(drop=True)
    probe = load_items(root, ["probe"], drop_invalid=False)[0]
    burn = pool[pool["split"] == "burn"]
    if frame_digest(burn) != fz["burn_frame_sha256"]:
        raise SystemExit("refusing to run the test step: the burn rows changed after the freeze (burn_frame_sha256 mismatch)")
    w_primary, merged = merge_weights(pool, fz["pi"])
    pool["w_primary"] = w_primary
    test = pool[pool["split"] == "test"].reset_index(drop=True)
    pool = pool.reset_index(drop=True)
    if not len(test) or not len(burn):
        raise SystemExit(f"refusing to run the test step: {len(burn)} usable burn rows, {len(test)} usable test rows (run ar3_features.py --split all first)")
    test_w = test["w_primary"].to_numpy(float)
    if merged:
        dev(f"families merged for the primary Q (fewer than 8 test items in a kind): {merged}")
    print(f"test: {len(test)} items, {test['entry_key'].nunique()} entries; pool {pool['entry_key'].nunique()} entries", flush=True)

    p7 = fz.get("p7")
    cfg = {"t": fz["t_star"], "p": fz["p_star"]}
    T = test.copy()
    T["weight"] = test_w
    A = arrays(T, p7)
    codes, groups = groups_of(T)
    E = len(groups)
    w, W = A["w"], A["w"].sum()
    hc, sc = wm(A["hc"], w), wm(A["sc"], w)
    cfg["e_eq"] = min(1.0, 2 * hc / sc)  # 3 Haiku passes + e Sonnet costs the same as 5 Haiku passes
    checks = replay_tests(A)
    print("replay unit tests (test):", checks, flush=True)

    # point estimates, bootstrap
    pt = compute(A, cfg)
    boot = boot_all(A, groups, cfg, B)
    # cross-check one statistic against ar3_stats.cluster_bootstrap (same seed, same draws)
    def xcheck():
        fr = T[["entry_key"]].copy()
        fr["i"] = np.arange(len(T))
        ref = S.cluster_bootstrap(fr, "entry_key", lambda f: compute(take(A, f["i"].to_numpy()), cfg)["G1"], B=50, seed=0)
        assert np.allclose(ref["boot"], boot["G1"].to_numpy()[:50], atol=1e-12), "local bootstrap draws differ from ar3_stats.cluster_bootstrap"
    xcheck()
    contrib = contributions(A, codes, E, cfg)
    n_items, n_ent = len(T), E
    kinds_n = T[T["kind"].isin(PLANTS)]["kind"].value_counts().reindex(PLANTS, fill_value=0)
    weights_rep = {"n_items": n_items, "n_entries": n_ent, "ess": float(W ** 2 / (w ** 2).sum()), "n_per_kind": kinds_n.to_dict(),
                   "n_controls": int((T["kind"] == "control").sum()), "min_items_per_kind": int(kinds_n.min()), "merged_families": merged}

    # gates
    rec, fa, per_kind = macro_recall_fa(T)
    # one entry per (item, model, pass): a failed call that a resumed run redid successfully counts once, as valid; dry-run records are skipped
    call_ok = {}
    for f in ("calls_burn.jsonl", "calls_test.jsonl"):
        p = root / f
        if p.exists():
            for ln in p.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict) or r.get("dry_run"):
                    continue
                key = (r.get("item_id"), r.get("model_alias"), r.get("pass_idx"))
                call_ok[key] = call_ok.get(key, False) or bool(r.get("ok"))
    calls_n, calls_ok = len(call_ok), sum(call_ok.values())
    valid_frac = calls_ok / calls_n if calls_n else float("nan")
    if not calls_n:
        dev("calls_burn.jsonl and calls_test.jsonl not found: valid-call gate cannot be assessed and fails")
    g0 = {"pool_entries": int(pool["entry_key"].nunique()), "valid_call_fraction": valid_frac, "n_calls": calls_n, "sonnet_macro_located_recall": rec,
          "sonnet_per_kind_recall": per_kind, "sonnet_fa": fa}
    g0["pass"] = bool(g0["pool_entries"] >= G0_POOL and valid_frac >= G0_VALID and rec >= G0_RECALL and fa <= G0_FA)
    G1_lb, G1_ub = lower95(boot["G1"]), upper95(boot["G1"])
    g1 = {"Q_S_minus_Q_H3": pt["G1"], "lb95": G1_lb, "ub95": G1_ub, "pass": bool(pt["G1"] >= G1_MIN and G1_lb > 0), "R2_vacuous": bool(G1_ub < M)}

    # reads at the frozen operating point
    e_hat = pt["e_hat"]
    s_fz = S.escalation_prob(A["agree"], cfg["t"], cfg["p"])
    n_items_esc = float(s_fz.sum())
    w_lo, w_hi = S.wilson(e_hat * g1_ess(w), g1_ess(w))
    i_lo, i_hi = S.wilson(n_items_esc, n_items)
    r1_perm = S.signflip_perm(contrib["R1"], n_sign, 0, "greater")
    r1 = {"Q_C": pt["Q_C"], "Q_rand_at_e_hat": pt["Q_rand_eh"], "diff": pt["R1"], "lb95": lower95(boot["R1"]), "perm_p": r1_perm["p"]}
    r1["pass"] = bool(pt["R1"] > 0 and r1["lb95"] > 0)
    d_r2 = contrib["R2"] - M * contrib["W"]
    r2_perm = S.signflip_perm(d_r2, n_sign, 0, "less")
    r2 = {"Q_S_minus_Q_C": pt["R2"], "ub95": upper95(boot["R2"]), "lb95": lower95(boot["R2"]), "margin": M, "perm_p_vs_margin": r2_perm["p"]}
    r2["pass"] = bool(pt["R2"] < M and r2["ub95"] < M)
    gap_ci = lo_hi(boot["Gap"], 0.90)
    k1 = bool(np.isfinite(gap_ci[1]) and gap_ci[1] < 0.25)
    k2 = bool(r2["lb95"] > M)
    n_gap_nan = int(boot["Gap"].isna().sum())
    if n_gap_nan:
        dev(f"gap closed undefined (oracle minus random < 0.01) in {n_gap_nan} of {B} resamples; those resamples are skipped in its interval")
    if not np.isfinite(pt["Gap"]):
        dev("gap closed at the frozen policy is undefined on the test slice (oracle minus random < 0.01)")
    decision = "PASS" if (g0["pass"] and g1["pass"] and r1["pass"] and r2["pass"]) else ("KILL" if (k1 or k2) else "INCONCLUSIVE")
    sd_r2 = float((contrib["R2"] * E).std(ddof=1))
    delta_r2 = abs(pt["R2"] - M)
    power = prow(sd_r2, E, delta_r2)

    # qualifier
    qual = {}
    for k, g in T.groupby("kind"):
        cw = (np.isclose(g["conf_agree"], 1.0) & (g["haiku_k3_correct"] == 0))
        agree_n = int(np.isclose(g["conf_agree"], 1.0).sum())
        qual[k] = {"n": int(len(g)), "confident_wrong_rate": float(cw.mean()), "confident_wrong_given_agree": float(cw.sum() / agree_n) if agree_n else float("nan")}
    except_kinds = [k for k, v in qual.items() if v["n"] >= 15 and v["confident_wrong_rate"] >= 0.5]

    # secondary family
    s_boot = {nm: boot[nm].to_numpy() for nm in ("S1", "S2", "S3") if nm in boot}
    p_s1 = float((1 + np.nansum(boot["S1"] <= 0)) / (B + 1))
    p_s3 = float((1 + np.nansum(boot["S3"] <= 0)) / (B + 1))
    p_s2 = float((1 + np.nansum(boot["S2"] >= 0.02)) / (B + 1)) if "S2" in boot else float("nan")
    ps = [p_s1, p_s2 if np.isfinite(p_s2) else 1.0, p_s3]
    holm = S.holm(ps)
    sec = {"S1_auroc_agree_minus_verb": {"est": pt["S1"], "lb95": lower95(boot["S1"]), "ci90": lo_hi(boot["S1"]), "p": p_s1, "holm_p": float(holm[0]),
                                         "auroc_agree": pt["auc_s_agree"], "auroc_verb": pt["auc_s_verb"]},
           "S2_P7_minus_P3_at_e25": {"est": pt.get("S2"), "ub95": upper95(boot["S2"]) if "S2" in boot else None, "ci90": lo_hi(boot["S2"]) if "S2" in boot else None,
                                     "p": p_s2, "holm_p": float(holm[1])},
           "S3_cascade_minus_P0c_equal_dollars": {"est": pt["S3"], "lb95": lower95(boot["S3"]), "ci90": lo_hi(boot["S3"]), "p": p_s3, "holm_p": float(holm[2]),
                                                  "e_equal_dollars": cfg["e_eq"]}}
    if p7 is None:
        dev("P7 missing from frozen.json (single-class burn error label): S2 is not evaluated and enters Holm with p = 1")

    # curves and policy table
    curves = policy_curves(A)
    ptab = policy_table(A, curves, hc, sc, pt["Q_S"])
    endpoints = {"P0a_haiku_k1": (cost_ratio(1, 0, hc, sc), pt["Q_H1"]), "P0b_haiku_k3": (cost_ratio(3, 0, hc, sc), pt["Q_H3"]),
                 "P0c_haiku_k5": (cost_ratio(5, 0, hc, sc), pt["Q_H5"]), "P1_sonnet": (1.0, pt["Q_S"]), "P1b_sonnet_maj3": (3.0, pt["Q_Smaj"])}

    # A/A noise: Sonnet pass pairs, paired bootstrap of |Q_a - Q_b|
    aa = {}
    pc = [c for c in ("sonnet_pass_correct_1", "sonnet_pass_correct_2", "sonnet_pass_correct_3") if c in T.columns and T[c].notna().all()]
    if len(pc) == 3:
        Qp = lambda c, ix: wm(T[c].to_numpy(float)[ix], w[ix])
        pairs = [(0, 1), (0, 2), (1, 2)]
        point = [abs(Qp(pc[a], slice(None)) - Qp(pc[b], slice(None))) for a, b in pairs]
        rng = np.random.default_rng(0)
        d = []
        for _ in range(B_sens):
            ix = np.concatenate([groups[j] for j in rng.integers(0, E, E)])
            d += [abs(Qp(pc[a], ix) - Qp(pc[b], ix)) for a, b in pairs]
        aa = {"sonnet_pair_point_abs_diffs": point, "sonnet_pair_boot_p95": float(np.percentile(d, 95)),
              "sonnet_pass_agreement": float(np.mean([wm((T[pc[a]] == T[pc[b]]).to_numpy(float), w) for a, b in pairs]))}
        dev("A/A noise floor is computed for Sonnet pass pairs only: the parquet has no per-pass Haiku correctness, so Haiku triples other than 1,2,3 cannot be scored")
    aa_band = aa.get("sonnet_pair_boot_p95", 0.0)

    # cost bookkeeping
    cost = {"haiku_per_call_weighted": hc, "sonnet_per_call_weighted": sc,
            "cascade_ratio_vs_P1": cost_ratio(3, e_hat, hc, sc), "P0b_ratio": cost_ratio(3, 0, hc, sc), "P0c_ratio": cost_ratio(5, 0, hc, sc)}

    # diagnostic block (no read): P3c at b*, diagnose table on test
    bstar = fz["b_star"]
    t_b, p_b = fz["secondary"]["conf_agree_at_bstar"]["t"], fz["secondary"]["conf_agree_at_bstar"]["p"]
    s_b = S.escalation_prob(A["agree"], t_b, p_b)
    diag_test = {}
    try:
        rep, _ = replicate(T)
        rep = rep.iloc[np.random.default_rng(1000).permutation(len(rep))].reset_index(drop=True)
        dsum = diag_summary(run_diagnose(rep, ["conf_agree"], "proxy_error_sonnet", root / "diag" / "test_s00"))["conf_agree"]
        diag_test = {"threshold_error_at_budgets": dsum["budgets"], "random_error_at_budgets": dsum["random"], "auc": dsum["auc"], "regime": dsum["regime"]}
    except SystemExit as ex:
        dev(f"diagnose on the test frame failed: {str(ex)[:120]}")
    diagnostic = {"b_star": bstar, "P3c_e_hat": wm(s_b, w), "P3c_Q": wm(A["qH3"] + s_b * (A["qS"] - A["qH3"]), w), "diagnose_test": diag_test}

    # ------------------------------------------------ exploratory
    ex = {}
    gain = A["qS"] - A["qH3"]
    pk = lambda key, base="qH3", e=E0: wm(A[base] + esc_probs(A[key], w, e) * (A["qS"] - A[base]), w)
    ex["k_sweep"] = {"k1": {"Q_cheap": pt["Q_H1"], "Q_cascade_e25_verb1": pk("verb1", "qH1"), "cost_ratio": cost_ratio(1, E0, hc, sc)},
                     "k3": {"Q_cheap": pt["Q_H3"], "Q_cascade_e25_agree": pk("agree"), "cost_ratio": cost_ratio(3, E0, hc, sc)},
                     "k5": {"Q_cheap": pt["Q_H5"], "Q_cascade_e25_agree10": pk("agree10", "qH5"), "cost_ratio": cost_ratio(5, E0, hc, sc)}}
    sv = S.escalation_prob(A["vote"], fz["secondary"]["conf_vote_e25"]["t"], fz["secondary"]["conf_vote_e25"]["p"])
    ex["P3b_vote_frozen"] = {"e_hat": wm(sv, w), "Q": wm(A["qH3"] + sv * gain, w)}
    ex["P5_escalate_on_flag"] = {"e_hat": wm(A["flag"], w), "Q": wm(A["qH3"] + A["flag"] * gain, w), "cost_ratio": cost_ratio(3, wm(A["flag"], w), hc, sc)}
    ex["P8"] = {"longest_Q_e25": pk("len"), "most_problems_Q_e25": pk("npr"), "random_Q_e25": pt["Q_H3"] + E0 * pt["G1"]}
    ex["per_kind_recall"] = {k: {"n": int(len(g)), "haiku_k1": float(g["haiku_k1_correct"].mean()), "haiku_k3": float(g["haiku_k3_correct"].mean()),
                                 "haiku_k5": float(g["haiku_k5_correct"].mean()), "sonnet_single": float(g["sonnet_single_expect"].mean())} for k, g in T.groupby("kind")}
    dev("tolerance sensitivity (0, +-3, any flag) is not in this run: items.parquet carries the +-1 tolerance only; rerun ar3_features.py with --tol into another run dir and rerun the test step on a copy")
    n10 = int(((T["haiku_k3_correct"] == 1) & (T["sonnet_majority_correct"] == 0)).sum())
    n01 = int(((T["haiku_k3_correct"] == 0) & (T["sonnet_majority_correct"] == 1)).sum())
    ex["mcnemar_haiku3_vs_sonnet_majority_items"] = {"n10": n10, "n01": n01, "mid_p": S.mcnemar_midp(n10, n01), "note": "item level, unweighted, clusters ignored"}
    # pi sensitivity: reweight plant share of the slice to pi'
    sens_pi = {}
    plant_mask = T["kind"].isin(PLANTS).to_numpy()
    share = float(w[plant_mask].sum() / W)
    for pi2 in (0.10, 0.25, 0.50):
        A2 = dict(A)
        A2["w"] = np.where(plant_mask, w * pi2 / share, w * (1 - pi2) / (1 - share))
        b2 = boot_all(A2, groups, cfg, B_sens)
        p2 = compute(A2, cfg)
        sens_pi[str(pi2)] = {"Q_S": p2["Q_S"], "Q_C": p2["Q_C"], "R1": p2["R1"], "R2": p2["R2"], "R2_ub95": upper95(b2["R2"]), "R1_lb95": lower95(b2["R1"]), "e_hat": p2["e_hat"],
                             "G1": p2["G1"]}
    ex["pi_sensitivity"] = sens_pi
    ex["M_sensitivity"] = {str(m): {"R2_pass": bool(pt["R2"] < m and r2["ub95"] < m), "R2_kill": bool(r2["lb95"] > m)} for m in (0.03, 0.05, 0.08)}
    # natural probes
    pr = {}
    if len(probe):
        for tr, g in probe.groupby(probe["truth"].fillna("none") if "truth" in probe else probe["kind"]):
            pr[str(tr)] = {"n": int(len(g)), "kind_counts": g["kind"].value_counts().to_dict(), "haiku_k3_flag_rate": float(g["haiku_k3_flag"].mean()),
                           "haiku_k3_correct_rate": float(g["haiku_k3_correct"].dropna().mean()) if g["haiku_k3_correct"].notna().any() else None,
                           "conf_agree_mean": float(g["conf_agree"].mean())}
    ex["natural_probes"] = pr
    # 1,000 random entry-level 25/75 splits of the pooled entries
    P = pool.copy()
    AP = arrays(P, None)
    cp, gp = groups_of(P)
    Ep = len(gp)
    rng = np.random.default_rng(0)
    rs = []
    nb = math.ceil(0.25 * Ep)
    for _ in range(n_splits):
        perm = rng.permutation(Ep)
        ib = np.concatenate([gp[j] for j in perm[:nb]])
        it = np.concatenate([gp[j] for j in perm[nb:]])
        Ab, At = take(AP, ib), take(AP, it)
        t_, p_ = S.weighted_quantile(Ab["agree"], Ab["w"], E0)
        st = S.escalation_prob(At["agree"], t_, p_)
        qc = wm(At["qH3"] + st * (At["qS"] - At["qH3"]), At["w"])
        rs.append({"b": b_star_analytic(Ab["agree"], Ab["pe_s"], Ab["w"]), "e_hat": wm(st, At["w"]), "QS_minus_QC": wm(At["qS"], At["w"]) - qc})
    rs = pd.DataFrame(rs)
    ex["random_splits"] = {"n": n_splits, "b_star_pct": np.percentile(rs["b"], [5, 50, 95]).tolist(), "e_hat_pct": np.percentile(rs["e_hat"], [5, 50, 95]).tolist(),
                           "QS_minus_QC_pct": np.percentile(rs["QS_minus_QC"], [5, 50, 95]).tolist(), "share_meeting_NI": float((rs["QS_minus_QC"] < M).mean())}
    # Trust-or-Escalate fixed-sequence scan (unweighted counts vs truth)
    alpha = fz["burn_sonnet_error_rate_unweighted"]
    wrong = 1 - T["haiku_k3_correct"].to_numpy(float)
    keep = None
    for t_ in sorted(np.unique(A["agree"]), reverse=True):
        acc = A["agree"] >= t_
        kk, nn = int(wrong[acc].sum()), int(acc.sum())
        ub = S.clopper_pearson(kk, nn, 0.05, "upper")[1]
        if ub <= alpha:
            keep = {"t": float(t_), "accepted_items": nn, "errors": kk, "cp_upper95": ub}
        else:
            break
    toe = {"alpha": alpha, "result": keep}
    if keep:
        st = (A["agree"] < keep["t"]).astype(float)
        toe.update(e_hat=wm(st, w), Q=wm(A["qH3"] + st * gain, w), Q_random_same_e=pt["Q_H3"] + wm(st, w) * pt["G1"])
    ex["trust_or_escalate"] = toe
    # oracle vs extra stratum
    if "stratum" in T.columns:
        strat = {}
        for sname, g in T.groupby("stratum"):
            ix = g.index.to_numpy()
            c = compute(take(A, ix), cfg)
            strat[str(sname)] = {"entries": int(g["entry_key"].nunique()), "items": int(len(g)), **{k: c[k] for k in ("Q_S", "Q_H3", "Q_C", "R1", "R2", "e_hat", "G1")}}
        ex["strata"] = strat
    ex["gap_closed_rank_at_e25"] = {"est": pt["Gap@0.25"], "ci90": lo_hi(boot["Gap@0.25"])}
    ex["grid_cis"] = {str(e): {"R1": {"est": pt[f"R1@{e}"], "ci90": lo_hi(boot[f"R1@{e}"])}, "R2": {"est": pt[f"R2@{e}"], "ci90": lo_hi(boot[f"R2@{e}"])}} for e in E_GRID}

    # ------------------------------------------------ outputs
    ent = f"{E} ({n_items})"
    pt_map = {"G1": pt["G1"], "R1": pt["R1"], "R2": pt["R2"], "S3": pt["S3"], "S2": pt.get("S2", 0.0)}
    pw = lambda name, boundary: prow(float((contrib[name] * E).std(ddof=1)), E, abs(pt_map[name] - boundary))
    fmt = lambda x: f"{x:+.3f}"
    ci = lambda nm: f"{fmt(pt[nm])} [{boot[nm].quantile(.05):+.3f}, {boot[nm].quantile(.95):+.3f}]"
    qm = lambda row: f"q={row['q']:.2f}, MDE={row['mde']:.3f}"
    r_g1, r_r1, r_r2 = pw("G1", G1_MIN), pw("R1", 0.0), pw("R2", M)
    rows = []
    rows.append(("G1 headroom Q_S - Q_H3 (>= 0.06)", ent, ci("G1"), f"{S.signflip_perm(contrib['G1'], n_sign, 0, 'greater')['p']:.4f}", "-", qm(r_g1),
                 f"{cost['P0b_ratio']:.2f}", "supported" if g1["pass"] else ("not supported" if G1_ub < G1_MIN else "unresolved")))
    rows.append(("R1 cascade - random at e_hat (> 0)", ent, ci("R1"), f"{r1['perm_p']:.4f}", "-", qm(r_r1), f"{cost['cascade_ratio_vs_P1']:.2f}",
                 verdict_ge(r1["lb95"], upper95(boot["R1"]), 0.0)))
    rows.append((f"R2 Q_S - cascade (< {M})", ent, ci("R2"), f"{r2['perm_p_vs_margin']:.4f}", "-", qm(r_r2), f"{cost['cascade_ratio_vs_P1']:.2f}",
                 verdict_lt(r2["lb95"], r2["ub95"], M)))
    ci_gap = f"{pt['Gap']:+.3f} [{gap_ci[0]:+.3f}, {gap_ci[1]:+.3f}]" if np.isfinite(gap_ci[1]) and np.isfinite(pt["Gap"]) else "undefined"
    rows.append(("K1 gap closed at frozen policy (supported if lower end >= 0.25, killed if upper end < 0.25)", ent, ci_gap, "-", "-", "-", f"{cost['cascade_ratio_vs_P1']:.2f}",
                 "not supported" if k1 else ("supported" if np.isfinite(gap_ci[0]) and gap_ci[0] >= 0.25 else "unresolved")))
    s1 = sec["S1_auroc_agree_minus_verb"]
    rows.append(("S1 AUROC conf_agree - conf_verb (> 0)", ent, ci("S1"), "-", f"{s1['holm_p']:.4f}", "-", "-",
                 "supported" if s1["holm_p"] < 0.05 else ("not supported" if upper95(boot["S1"]) <= 0 else "unresolved")))
    if "S2" in boot:
        s2 = sec["S2_P7_minus_P3_at_e25"]
        rows.append(("S2 P7 - P3 at e=0.25 (< 0.02)", ent, ci("S2"), "-", f"{s2['holm_p']:.4f}", qm(pw("S2", 0.02)), f"{cost_ratio(3, E0, hc, sc):.2f}",
                     "supported" if s2["holm_p"] < 0.05 else ("not supported" if lower95(boot["S2"]) >= 0.02 else "unresolved")))
    s3 = sec["S3_cascade_minus_P0c_equal_dollars"]
    rows.append(("S3 cascade - all-Haiku k=5 at equal dollars (> 0)", ent, ci("S3"), f"{S.signflip_perm(contrib['S3'], n_sign, 0, 'greater')['p']:.4f}", f"{s3['holm_p']:.4f}",
                 qm(pw("S3", 0.0)), f"{cost['P0c_ratio']:.2f}", "supported" if s3["holm_p"] < 0.05 else ("not supported" if upper95(boot["S3"]) <= 0 else "unresolved")))
    md = ["| contrast | entries (items) | paired diff with 90% CI (5th to 95th pct) | permutation p | Holm p | q and MDE | cost ratio vs all-Sonnet | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    head = (f"AR3 decision: **{decision}**" + (" (unresolved: see MDE and q)" if decision == "INCONCLUSIVE" else "") +
            f". G0 {'pass' if g0['pass'] else 'FAIL'}, G1 {'pass' if g1['pass'] else 'FAIL'}, R1 {'pass' if r1['pass'] else 'FAIL'}, R2 {'pass' if r2['pass'] else 'FAIL'}, K1 {k1}, K2 {k2}. "
            f"Realized e_hat {e_hat:.3f}." + (f" Claim worded except kind {', '.join(except_kinds)}." if except_kinds else ""))
    (root / "table.md").write_text(head + "\n\n" + "\n".join(md) + "\n")

    res = {"decision": decision, "k1": k1, "k2": k2, "except_kinds": except_kinds, "gates": {"G0": g0, "G1": g1}, "reads": {"R1": r1, "R2": r2},
           "gap_closed": {"est": pt["Gap"], "ci90": gap_ci, "n_undefined_resamples": n_gap_nan},
           "e_hat": {"weighted": e_hat, "wilson_at_ess": [w_lo, w_hi], "items": n_items_esc / n_items, "wilson_items": [i_lo, i_hi], "boot_ci90": lo_hi(boot["e_hat"]), "target": E0},
           "power": {"R2": power, "R1": r_r1, "G1": r_g1}, "secondary": sec, "qualifier": qual, "point": pt, "weights": weights_rep, "cost": cost,
           "policies": ptab, "endpoints": endpoints, "aa_noise": aa, "diagnostic": diagnostic, "exploratory": ex, "replay_unit_tests": checks,
           "frozen": {k: fz[k] for k in ("t_star", "p_star", "b_star", "b_star_interval", "items_sha256", "split_sha256", "system_sha256", "schema_sha256")},
           "model_ids_seen": {"burn": fz.get("model_ids_seen"), "test": models_seen(root, ["calls_test.jsonl"])},
           "per_item_cost": ({str(r.item_id): {"haiku": float(r.haiku_cost_mean), "sonnet": float(r.sonnet_cost_mean)} for r in T.itertuples()} if "item_id" in T.columns else None),
           "per_entry_cost": {k: {"haiku": float(g["haiku_cost_mean"].mean()), "sonnet": float(g["sonnet_cost_mean"].mean())} for k, g in T.groupby("entry_key")},
           "boot": {"B": B, "seed": 0, "perm": n_sign}, "deviations": DEVIATIONS, "seconds": time.time() - t0}
    wjson(root / "result.json", res)
    wjson(root / "cascade_config.json", {"k": K, "system_sha256": fz.get("system_sha256"), "schema_sha256": fz.get("schema_sha256"), "t_star": cfg["t"], "p_star": cfg["p"],
                                          "b_star": bstar, "e_primary": E0, "dollars_per_call": {"haiku": hc, "sonnet": sc}})
    test_figs(root, T, A, curves, endpoints, aa_band, pt, fz, qual, boot, cfg, (hc, sc), pool, burn)
    print(head)
    print("\n".join(md))
    print(f"wrote {root / 'result.json'}, table.md, cascade_config.json, fig/ in {time.time() - t0:.0f}s")
    return 0


def g1_ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


# ---------------------------------------------------------------- figures


def test_figs(root, T, A, curves, endpoints, aa_band, pt, fz, qual, boot, cfg, costs, pool, burn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fd = root / "fig"
    fd.mkdir(parents=True, exist_ok=True)
    hc, sc = costs
    w = A["w"]
    col = {"P3_agree": OI["blue"], "P3b_vote": OI["sky"], "P4a_verb1": OI["orange"], "P4b_verb": OI["red"], "P7_learned": OI["green"], "P8_longest": OI["purple"],
           "P8_most_problems": OI["grey"], "P6_oracle": OI["black"], "P2_random": OI["grey"]}
    # cost-quality
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhspan(pt["Q_S"] - aa_band, pt["Q_S"] + aa_band, color=OI["grey"], alpha=0.15, label=f"A/A noise band (Sonnet pass pairs, p95 {aa_band:.3f})")
    pts = []
    for nm, c in curves.items():
        x = np.array([cost_ratio(c["k"], e, hc, sc) for e in FINE])
        ax.plot(x, c["q"], color=col.get(nm, OI["grey"]), lw=2 if nm in ("P3_agree", "P6_oracle") else 1.2, ls="--" if nm in ("P2_random", "P6_oracle") else "-", label=nm)
        pts += [(float(a), float(b), nm) for a, b in zip(x, c["q"])]
    xmax = max(max(cost_ratio(c["k"], 1.0, hc, sc) for c in curves.values()), 1.0) * 1.08
    for nm, (x, y) in endpoints.items():
        if x > xmax:
            ax.text(xmax, y, f"{nm.split('_')[0]} at cost {x:.1f} (off axis)", ha="right", va="bottom", fontsize=7)
            continue
        ax.scatter([x], [y], color=OI["black"], zorder=5, s=22)
        ax.annotate(nm.split("_")[0], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        pts.append((x, y, nm))
    pf = pareto(pts)
    ax.step([p[0] for p in pf], [p[1] for p in pf], where="post", color=OI["red"], lw=0.8, alpha=0.7, label="Pareto frontier")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("dollar cost relative to all-Sonnet single pass")
    ax.set_ylabel("prevalence-weighted Q")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fd / "cost_quality.png", dpi=140)
    plt.close(fig)
    # risk-coverage
    fig, ax = plt.subplots(figsize=(6, 4.2))
    err = 1 - A["qH3"]
    cov = np.linspace(0.05, 1, 96)
    for nm, key, c in (("conf_agree", A["agree"], OI["blue"]), ("conf_vote", A["vote"], OI["sky"]), ("conf_verb", A["verb"], OI["red"]), ("random", np.zeros(len(w)), OI["grey"])):
        r = []
        for cv in cov:
            s = esc_probs(key, w, 1 - cv)
            r.append(float((w * (1 - s) * err).sum() / (w * (1 - s)).sum()))
        ax.plot(cov, r, label=nm, color=c)
    ax.set_xlabel("coverage (weight fraction accepted without escalation)")
    ax.set_ylabel("Haiku k=3 error on accepted set (vs truth)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "risk_coverage.png", dpi=140)
    plt.close(fig)
    # AUROC bars
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    bb = {c: S.weighted_auroc(burn["proxy_error_sonnet"].to_numpy(float), -burn[c].to_numpy(float), burn["weight"].to_numpy(float)) for c in SIGNALS}
    bt = {c: S.weighted_auroc(burn["proxy_error_truth"].to_numpy(float), -burn[c].to_numpy(float), burn["weight"].to_numpy(float)) for c in SIGNALS}
    for a, tgt, bv in zip(axs, ("s", "t"), (bb, bt)):
        x = np.arange(4)
        a.bar(x - 0.2, [bv[c] for c in SIGNALS], 0.4, color=OI["sky"], label="burn")
        a.bar(x + 0.2, [pt[f"auc_{tgt}_{c.replace('conf_', '')}"] for c in SIGNALS], 0.4, color=OI["blue"], label="test")
        a.axhline(0.5, color=OI["grey"], lw=0.8, ls=":")
        a.set_xticks(x)
        a.set_xticklabels([c.replace("conf_", "") for c in SIGNALS])
        a.set_title("target: " + ("proxy_error_sonnet" if tgt == "s" else "proxy_error_truth"), fontsize=9)
    axs[0].set_ylabel("weighted AUROC (low confidence predicts error)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "auroc_bars.png", dpi=140)
    plt.close(fig)
    # reliability
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0001])
    for a, key, nm in zip(axs, (A["agree"], A["verb"]), ("conf_agree", "conf_verb")):
        b = np.digitize(key, edges) - 1
        xs, ys, ns = [], [], []
        for i in range(5):
            m = b == i
            if m.any():
                xs.append(float(np.average(key[m], weights=w[m]))); ys.append(wm(A["qH3"][m], w[m])); ns.append(int(m.sum()))
        a.plot([0, 1], [0, 1], color=OI["grey"], ls=":")
        a.scatter(xs, ys, s=[min(20 + 3 * n, 500) for n in ns], color=OI["blue"])
        a.set_title("marker size grows with items in bin", fontsize=8)
        a.set_xlabel(nm)
        a.set_xlim(0, 1.02)
        a.set_ylim(0, 1.02)
    axs[0].set_ylabel("weighted Haiku k=3 accuracy")
    fig.tight_layout()
    fig.savefig(fd / "reliability.png", dpi=140)
    plt.close(fig)
    # histograms
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, key, nm in zip(axs, (A["agree"], A["verb"]), ("conf_agree", "conf_verb")):
        bins = np.linspace(0, 1, 21)
        a.hist([key[A["qH3"] == 1], key[A["qH3"] == 0]], bins=bins, stacked=True, color=[OI["sky"], OI["red"]], label=["Haiku k=3 right", "wrong"])
        a.set_xlabel(nm)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "conf_histograms.png", dpi=140)
    plt.close(fig)
    # confident-wrong by kind
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ks = list(qual)
    ax.bar(ks, [qual[k]["confident_wrong_rate"] for k in ks], color=OI["red"])
    ax.axhline(0.5, color=OI["black"], lw=0.8, ls="--")
    for i, k in enumerate(ks):
        ax.text(i, qual[k]["confident_wrong_rate"] + 0.01, f"n={qual[k]['n']}", ha="center", fontsize=7)
    ax.set_ylabel("confident-wrong rate")
    fig.tight_layout()
    fig.savefig(fd / "confident_wrong_by_kind.png", dpi=140)
    plt.close(fig)
    # burn-in predicted vs realized e
    bp = root / "burn_bootstrap.csv"
    if bp.exists():
        bt_ = pd.read_csv(bp)
        real = [wm(S.escalation_prob(A["agree"], r.t, r.p), w) for r in bt_.itertuples()]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(real, bins=20, color=OI["sky"], edgecolor="white", label="test e under each burn-bootstrap (t, p)")
        ax.axvline(E0, color=OI["black"], ls="--", label="burn-in target 0.25")
        ax.axvline(pt["e_hat"], color=OI["red"], label=f"realized at frozen (t*, p*): {pt['e_hat']:.3f}")
        ax.set_xlabel("escalation rate on the test slice")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fd / "burn_predicted_vs_realized_e.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------- selftest


def fake_items(scn, seed=0, n_entries=64):
    """Controls and the five plant kinds per entry. Haiku is wrong on 15 percent of items. 'informative': conf_agree is 1.0 exactly when
    Haiku is right; 'random': conf_agree is independent of correctness. Sonnet is always right."""
    rng = np.random.default_rng(seed)
    ents = [f"e{j:03d}" for j in range(n_entries)]
    order = sorted(ents, key=lambda e: hashlib.sha256((e + "ar3-selftest").encode()).hexdigest())
    nb = math.ceil(0.25 * n_entries)
    split = {e: ("burn" if i < nb else "test") for i, e in enumerate(order)}
    rows = []
    for e in ents:
        for kind in ["control"] + PLANTS:
            wrong = rng.random() < 0.15
            ag = (1.0 if not wrong else float(rng.choice([0.0, 1 / 3, 0.5, 2 / 3]))) if scn == "informative" else float(rng.choice([0.0, 1 / 3, 0.5, 2 / 3, 1.0]))
            h = float(not wrong)
            rows.append({"item_id": f"{e}-{kind}", "entry_key": e, "stratum": "oracle" if int(e[1:]) % 2 else "extra", "split": split[e], "kind": kind,
                         "log_chars": 7.0 + rng.normal(0, 0.3), "n_numeric_tokens": 30, "has_threshold_language": int(rng.random() < 0.5),
                         "conf_agree": ag, "conf_agree_10": ag, "conf_vote": float(rng.choice([2 / 3, 1.0])), "conf_verb": float(np.clip(0.9 - 0.03 * wrong + rng.normal(0, 0.05), 0, 1)),
                         "conf_verb1": float(rng.uniform(0.5, 1)), "mean_n_problems": float(rng.integers(0, 3)), "flag_frac": float(rng.choice([0, 1 / 3, 1.0])), "q_ok": 1.0,
                         "h1_correct": float(h if rng.random() < 0.9 else 1 - h), "h3_correct": h, "h5_correct": h, "h3_flag": float((kind == "control") == wrong),
                         "s1_correct": 1.0, "s2_correct": 1.0, "s3_correct": 1.0, "sonnet_single_expect": 1.0, "sonnet_majority_correct": 1.0,
                         "proxy_error_sonnet": float(wrong), "proxy_error_truth": float(wrong), "haiku_cost_mean_usd": 0.004 + rng.uniform(0, 0.001),
                         "sonnet_cost_mean_usd": 0.03 + rng.uniform(0, 0.005), "valid": True})
    for i in range(3):
        rows.append({**rows[0], "item_id": f"p{i}", "entry_key": f"probe{i}", "split": "probe", "kind": "control", "truth": "unknown"})
    df = pd.DataFrame(rows)
    pool = df[df["split"].isin(["burn", "test"])]
    nc = (pool["kind"] == "control").sum()
    nk = pool["kind"].value_counts()
    df["w"] = [0.0 if s == "probe" else ((1 - PI) / nc if k == "control" else (PI / 5) / nk[k]) for s, k in zip(df["split"], df["kind"])]
    return df, split


def selftest(keep=None):
    S._tests()
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
    with (contextlib.nullcontext(str(keep)) if keep else tempfile.TemporaryDirectory(prefix="ar3_selftest_")) as td:
        out = {}
        for scn in ("informative", "random"):
            root = Path(td) / scn
            root.mkdir()
            df, split = fake_items(scn, seed=1 if scn == "random" else 0)
            df.to_parquet(root / "items.parquet", index=False)
            wjson(root / "split.json", split)
            (root / "items.sha256").write_text("0" * 64 + "  items.jsonl\n")
            calls = [{"item_id": f"x{i}", "pass_idx": 1, "ok": True, "model_alias": "haiku", "model_ids": {"claude-haiku-x": {}}} for i in range(40)] + \
                    [{"item_id": f"y{i}", "pass_idx": 1, "ok": True, "model_alias": "sonnet", "model_ids": {"claude-sonnet-x": {}}} for i in range(20)]
            calls += [{**calls[0], "ok": False}, {"item_id": "dry", "pass_idx": 1, "ok": False, "model_alias": "haiku", "dry_run": True}]  # a failed call later redone, and a dry-run record
            for f in ("calls_burn.jsonl", "calls_test.jsonl"):
                (root / f).write_text("\n".join(json.dumps(c) for c in calls) + "\n")
            # the test step must refuse before a freeze
            try:
                step_test(root, B=20, n_sign=99, n_splits=10, B_sens=20)
                raise AssertionError("test step ran without a freeze")
            except SystemExit as e:
                assert "frozen" in str(e)
            assert step_burn(root, n_shuffle=3, n_boot_burn=100) == 0
            fz = json.loads((root / "frozen.json").read_text())
            assert fz["burn_auroc_vs_sonnet"]["conf_agree"] > (0.99 if scn == "informative" else 0.0)
            if scn == "informative":
                assert fz["burn_auroc_vs_sonnet"]["conf_agree"] == 1.0, fz["burn_auroc_vs_sonnet"]
                assert abs(fz["t_star"] - 1.0) < 1e-9 and 0 < fz["p_star"] < 1
                assert fz["b_star"] <= 0.25
            else:
                assert abs(fz["burn_auroc_vs_sonnet"]["conf_agree"] - 0.5) < 0.3
            # tamper check: edit frozen.json without the hash
            raw = (root / "frozen.json").read_text()
            (root / "frozen.json").write_text(raw.replace('"k": 3', '"k": 4'))
            try:
                step_test(root, B=20, n_sign=99, n_splits=10, B_sens=20)
                raise AssertionError("test step ran with a tampered freeze")
            except SystemExit as e:
                assert "sha256" in str(e)
            (root / "frozen.json").write_text(raw)
            # a post-freeze relabel changes every control weight (N_controls moves): the burn digest must not trip on it
            d2 = pd.read_parquet(root / "items.parquet")
            d2.loc[d2["kind"] == "control", "w"] *= 1.1
            d2.to_parquet(root / "items.parquet", index=False)
            assert step_test(root, B=300, n_sign=999, n_splits=50, B_sens=60) == 0
            res = json.loads((root / "result.json").read_text())
            for fn in ("cost_quality", "risk_coverage", "auroc_bars", "reliability", "conf_histograms", "confident_wrong_by_kind", "burn_predicted_vs_realized_e"):
                assert (root / "fig" / f"{fn}.png").stat().st_size > 1000, fn
            assert (root / "table.md").exists() and (root / "cascade_config.json").exists()
            assert res["gates"]["G0"]["valid_call_fraction"] == 1.0 and res["gates"]["G0"]["n_calls"] == 60, res["gates"]["G0"]  # redone and dry-run calls do not count against the gate
            assert len(res["per_item_cost"]) == res["weights"]["n_items"]
            pol = res["policies"]
            if scn == "informative":
                # a perfectly informative signal sits on the oracle curve wherever the oracle can still gain (e <= error mass)
                a, o = pol["P3_agree"]["Q_at_grid"], pol["P6_oracle"]["Q_at_grid"]
                assert all(abs(a[k] - o[k]) < 1e-9 for k in a), (a, o)
                assert res["decision"] == "PASS", res["decision"]
                assert abs(pol["P3_agree"]["gap_closed_at_grid"]["0.1"] - 1.0) < 1e-6
                assert res["reads"]["R2"]["Q_S_minus_Q_C"] < 1e-9
            else:
                assert res["decision"] != "PASS"
                rnd, a = pol["P2_random"]["Q_at_grid"], pol["P3_agree"]["Q_at_grid"]
                assert abs(a["0.4"] - rnd["0.4"]) < 0.08, (a, rnd)
                assert res["secondary"]["S1_auroc_agree_minus_verb"]["est"] < 0.2
            assert res["replay_unit_tests"]["oracle_maximal_everywhere"]
            out[scn] = {"decision": res["decision"], "e_hat": res["e_hat"]["weighted"], "R1": res["reads"]["R1"]["diff"], "R2": res["reads"]["R2"]["Q_S_minus_Q_C"]}
        # early-gate stop: Sonnet that never finds anything makes the burn step print NOT TESTABLE and not freeze
        root = Path(td) / "bad"
        root.mkdir()
        df, split = fake_items("informative")
        for c in ("s1_correct", "s2_correct", "s3_correct", "sonnet_single_expect", "sonnet_majority_correct"):
            df[c] = 0.0
        df.to_parquet(root / "items.parquet", index=False)
        assert step_burn(root, n_shuffle=1, n_boot_burn=10) == 3 and not (root / "frozen.json").exists()
        # tie handling: constant key escalates the same fraction everywhere
        s = esc_probs(np.zeros(5), np.ones(5), 0.3)
        assert np.allclose(s, 0.3)
        print("selftest results:", json.dumps(jclean(out)))
    print("ar3_analyze selftest: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description="AR3 burn-in/freeze and test analysis")
    ap.add_argument("--step", choices=["burn", "test"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(RUN), help="run directory (default runs/AR3)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--sign", type=int, default=9999)
    ap.add_argument("--n_shuffle", type=int, default=20)
    ap.add_argument("--keep", default=None, help="selftest: keep outputs in this directory instead of a temp dir")
    ap.add_argument("--force", action="store_true", help="burn step: overwrite an existing freeze")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.keep)
    if a.step == "burn":
        return step_burn(a.root, n_shuffle=a.n_shuffle, force=a.force)
    if a.step == "test":
        return step_test(a.root, B=a.boot, n_sign=a.sign)
    ap.error("--step or --selftest is required")


if __name__ == "__main__":
    sys.exit(main())
