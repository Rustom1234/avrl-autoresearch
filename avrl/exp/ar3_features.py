"""AR3: per-item features for the critic cascade (Haiku passes, agreement-based escalation to Sonnet).

Reads runs/AR3/items.jsonl (substrate) and calls_{burn,test,probe}.jsonl (runner), and writes
runs/AR3/items.parquet with one row per item: agreement and confidence signals from the Haiku
passes, the cheap aggregate decision for k in 1,3,5, correctness against truth, Sonnet per-pass
and majority correctness, the two proxy-error labels, and the section-7 weights.
Also writes the blind control review pile (runs/AR3/review_pile_<split>.csv).
See exp/AR3_DESIGN.md sections 6 to 8. The test split stays closed until runs/AR3/frozen.json
matches runs/AR3/frozen.sha256.

  python exp/ar3_features.py --split burn [--pile]
  python exp/ar3_features.py --split all
  python exp/ar3_features.py --selftest
"""
import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "AR3"

PI = 0.25
TOL = 1
PLANT_KINDS = ("N-big", "N-mid", "N-stale", "V-body", "V-coh")
K_PASSES = {1: (1,), 3: (1, 2, 3), 5: (1, 2, 3, 4, 5)}
LINE_RE = re.compile(r"^L(\d+)\| ?(.*)$")
NAN = float("nan")
MODEL_WORDS = re.compile(r"\b(?:claude|haiku|sonnet|opus|anthropic)\b", re.I)

# ---------------------------------------------------------------- pass level


def parse_pass(so):
    """Structured output -> {lines, probs, conf}, or None when the call is not a usable critic answer."""
    if not isinstance(so, dict) or not isinstance(so.get("problems"), list):
        return None
    probs = []
    for p in so["problems"]:
        ln = p.get("line") if isinstance(p, dict) else None
        if isinstance(ln, int) and not isinstance(ln, bool):
            probs.append((ln, str(p.get("quote") or ""), str(p.get("explanation") or "")))
    c = so.get("judgment_confidence")
    conf = float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None
    return {"lines": {ln for ln, _, _ in probs}, "probs": probs, "conf": conf}


def n_matched(a, b):
    """One-to-one greedy matching of line sets: exact matches first, then |difference| == 1 (lower neighbour first)."""
    a, b = sorted(a), set(b)
    used, n, rest = set(), 0, []
    for x in a:
        if x in b:
            used.add(x)
            n += 1
        else:
            rest.append(x)
    for x in rest:
        for y in (x - 1, x + 1):
            if y in b and y not in used:
                used.add(y)
                n += 1
                break
    return n


def sim(a, b):
    """1 if both empty, 0 if exactly one is empty, else Dice overlap with +-1 greedy matching."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * n_matched(a, b) / (len(a) + len(b))


def mean_pair_sim(sets):
    return float(np.mean([sim(a, b) for a, b in combinations(sets, 2)]))


def near(line, s):
    return any(abs(line - x) <= 1 for x in s)


def supported(sets, thresh):
    """Lines (among all reported lines) that at least `thresh` passes report within +-1."""
    cand = set().union(*sets) if sets else set()
    return {ln for ln in cand if sum(near(ln, s) for s in sets) >= thresh}


def lines_disagree(h_sup, s_sup):
    """Label-free proxy error: one side flags and the other does not, or both flag with no shared line within +-1."""
    if not h_sup and not s_sup:
        return False
    if not h_sup or not s_sup:
        return True
    return not any(abs(h - s) <= 1 for h in h_sup for s in s_sup)


def accept_set(plant, tol):
    """Stored accept_lines already carry the +-1 of design 3.8, so tol=1 is the stored set.
    tol>1 widens it by tol-1; tol=0 drops the +-1 neighbours of the planted lines (extras adjacent to a planted line are dropped too)."""
    acc = set(plant["accept_lines"])
    if tol == 1:
        return acc
    if tol > 1:
        return {ln + d for ln in acc for d in range(-(tol - 1), tol)}
    pl = plant["lines"]
    return {ln for ln in acc if ln in pl or all(abs(ln - p) > 1 for p in pl)}


def judge(truth, plant, sup, tol=TOL):
    """(flag, correct, flagged_elsewhere); correct and flagged_elsewhere are None when truth is not clean or a plant kind."""
    flag = bool(sup)
    if truth == "clean":
        return flag, not flag, False
    if truth in PLANT_KINDS:
        hit = flag if tol == "any" else bool(set(sup) & accept_set(plant, tol))
        return flag, hit, flag and not hit
    return flag, None, None


def lines_of(text):
    return {int(m.group(1)): (ln, m.group(2)) for ln, m in ((ln, LINE_RE.match(ln)) for ln in text.split("\n")) if m}


def quote_ok(lmap, line, quote):
    """Quote found verbatim (whitespace-normalised) in the rendered lines line-1..line+1."""
    q = " ".join(quote.split())
    win = [lmap[ln] for ln in (line - 1, line, line + 1) if ln in lmap]
    return bool(q) and any(q in " ".join(" ".join(x[i] for x in win).split()) for i in (0, 1))


def _f(v):
    return NAN if v is None else float(v)


# ---------------------------------------------------------------- item level


def features(it, rec, tol=TOL):
    hp, sp = rec["haiku"], rec["sonnet"]
    cov = it["covariates"]
    f = {k: NAN for k in ("conf_agree", "conf_agree_10", "conf_vote", "conf_verb", "conf_verb1", "mean_n_problems", "flag_frac", "q_ok",
                          "s1_correct", "s2_correct", "s3_correct", "sonnet_single_expect", "sonnet_majority_flag",
                          "sonnet_majority_correct", "sonnet_agree", "proxy_error_sonnet", "proxy_error_truth")}
    for k in K_PASSES:
        f.update({f"h{k}_flag": NAN, f"h{k}_correct": NAN, f"h{k}_flagged_elsewhere": NAN, f"h{k}_supported": []})
    f["sonnet_majority_lines"] = []
    f.update(n_haiku_valid=len(hp), n_sonnet_valid=len(sp), log_chars=math.log(max(cov["n_chars"], 1)),
             n_numeric_tokens=cov["n_numeric_tokens"], has_threshold_language=int(bool(cov["has_threshold_language"])))
    f["valid"] = all(i in hp for i in K_PASSES[3]) and len(sp) >= 2
    truth, plant = it["truth"], it["plant"]

    if all(i in hp for i in K_PASSES[3]):
        tri = [hp[i] for i in K_PASSES[3]]
        f["conf_agree"] = mean_pair_sim([p["lines"] for p in tri])
        fr = sum(bool(p["probs"]) for p in tri) / 3
        f["flag_frac"], f["conf_vote"] = fr, max(fr, 1 - fr)
        cs = [p["conf"] for p in tri if p["conf"] is not None]
        f["conf_verb"] = sum(cs) / len(cs) / 100 if cs else NAN
        f["conf_verb1"] = tri[0]["conf"] / 100 if tri[0]["conf"] is not None else NAN
        f["mean_n_problems"] = float(np.mean([len(p["probs"]) for p in tri]))
        lmap = lines_of(it["text"])
        qs = [quote_ok(lmap, ln, q) for p in tri for ln, q, _ in p["probs"]]
        f["q_ok"] = float(np.mean(qs)) if qs else 1.0  # no quotes to check: nothing unsupported
    if all(i in hp for i in K_PASSES[5]):
        f["conf_agree_10"] = float(np.mean([mean_pair_sim([hp[i]["lines"] for i in t]) for t in combinations(K_PASSES[5], 3)]))

    for k, idx in K_PASSES.items():
        if all(i in hp for i in idx):
            sup = supported([hp[i]["lines"] for i in idx], math.ceil(k / 2))
            flag, cor, els = judge(truth, plant, sup, tol)
            f.update({f"h{k}_flag": _f(flag), f"h{k}_correct": _f(cor), f"h{k}_flagged_elsewhere": _f(els), f"h{k}_supported": sorted(sup)})

    if sp:
        cors = []
        for i in (1, 2, 3):
            if i in sp:
                c = judge(truth, plant, supported([sp[i]["lines"]], 1), tol)[1]
                f[f"s{i}_correct"] = _f(c)
                cors.append(c)
        if cors and all(c is not None for c in cors):
            f["sonnet_single_expect"] = float(np.mean(cors))
        if len(sp) >= 2:
            sup = supported([p["lines"] for p in sp.values()], 2)
            flag, cor, _ = judge(truth, plant, sup, tol)
            f.update(sonnet_majority_flag=_f(flag), sonnet_majority_correct=_f(cor), sonnet_majority_lines=sorted(sup))
            f["sonnet_agree"] = mean_pair_sim([p["lines"] for p in sp.values()])
            if not math.isnan(f["h3_flag"]):
                f["proxy_error_sonnet"] = float(lines_disagree(set(f["h3_supported"]), sup))
    if not math.isnan(f["h3_correct"]):
        f["proxy_error_truth"] = 1.0 - f["h3_correct"]
    mean = lambda v: float(np.mean(v)) if v else NAN
    f["haiku_cost_mean_usd"], f["sonnet_cost_mean_usd"] = mean(rec["cost"]["haiku"]), mean(rec["cost"]["sonnet"])
    return f


# ---------------------------------------------------------------- loading


def load_control_labels(run=RUN):
    """{entry_key: "clean"|"natural"}; missing file means every control is clean."""
    p = Path(run) / "control_labels.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    bad = {k: v for k, v in d.items() if v not in ("clean", "natural")}
    if bad:
        raise ValueError(f"control_labels.json has values other than clean/natural: {list(bad.items())[:3]}")
    return d


def load_items(run=RUN):
    """items.jsonl with split (final), orig_split and truth added. Controls labeled natural move to the probe split."""
    labels = load_control_labels(run)
    items = [json.loads(ln) for ln in (Path(run) / "items.jsonl").read_text().splitlines() if ln.strip()]
    assert len({it["item_id"] for it in items}) == len(items), "duplicate item_id in items.jsonl"
    for it in items:
        it["orig_split"] = it["split"]
        if it["kind"] != "control":
            it["truth"] = it["kind"]
        elif it["orig_split"] == "probe":  # natural probes: truth comes later from the orchestrator
            it["truth"] = labels.get(it["entry_key"], "unknown")
        else:
            it["truth"] = labels.get(it["entry_key"], "clean")
            if it["truth"] == "natural":
                it["split"] = "probe"
    return items


def check_frozen(run=RUN):
    fj, fs = Path(run) / "frozen.json", Path(run) / "frozen.sha256"
    if not (fj.exists() and fs.exists()):
        raise SystemExit("refusing to open the test split: frozen.json or frozen.sha256 is missing")
    tok = fs.read_text().split()
    want = tok[0].lower() if tok else ""
    got = hashlib.sha256(fj.read_bytes()).hexdigest()
    if want != got:
        raise SystemExit(f"refusing to open the test split: frozen.json sha256 {got[:12]} != frozen.sha256 {want[:12]}")


def load_calls(run, files, ids):
    """{item_id: {haiku: {pass_idx: parsed}, sonnet: {...}, cost: {haiku: [...], sonnet: [...]}}} from calls_<f>.jsonl for f in files.
    A call is valid iff ok and structured_output has a problems list; the first valid record per (item, model, pass) wins."""
    if "test" in files:
        check_frozen(run)
    raw = {}
    for f in sorted(files):
        p = Path(run) / f"calls_{f}.jsonl"
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue  # truncated last line of a killed runner
            if r.get("item_id") not in ids or not r.get("ok"):
                continue
            alias = str(r.get("model_alias", "")).lower()
            alias = "haiku" if "haiku" in alias else "sonnet" if "sonnet" in alias else None
            parsed = parse_pass(r.get("structured_output"))
            if alias and parsed:
                raw.setdefault((alias, r["item_id"], int(r["pass_idx"])), (parsed, r.get("total_cost_usd")))
    shift = {a: int(any(k[0] == a and k[2] == 0 for k in raw)) for a in ("haiku", "sonnet")}  # tolerate 0-based pass_idx
    out = {}
    for (alias, iid, idx), (parsed, cost) in raw.items():
        e = out.setdefault(iid, {"haiku": {}, "sonnet": {}, "cost": {"haiku": [], "sonnet": []}})
        e[alias][idx + shift[alias]] = parsed
        if isinstance(cost, (int, float)):
            e["cost"][alias].append(float(cost))
    return out


EMPTY = lambda: {"haiku": {}, "sonnet": {}, "cost": {"haiku": [], "sonnet": []}}

# ---------------------------------------------------------------- weights (section 7)


def weights(items, pi=PI, groups=None):
    """item_id -> weight from the FULL burn+test pool: control (1-pi)/N_controls, plant (pi/5)/N_kind.
    `groups` maps kind -> group name to merge kinds; a group of m kinds with N items gets m*(pi/5)/N per item."""
    g = lambda k: (groups or {}).get(k, k)
    pool = [it for it in items if it["split"] in ("burn", "test") and (it["truth"] == "clean" or it["kind"] in PLANT_KINDS)]
    nc = sum(it["kind"] == "control" for it in pool)
    ng = Counter(g(it["kind"]) for it in pool if it["kind"] in PLANT_KINDS)
    m = Counter(g(k) for k in PLANT_KINDS)
    w = {it["item_id"]: 0.0 for it in items}
    for it in pool:
        if it["kind"] == "control":
            w[it["item_id"]] = (1 - pi) / nc
        else:
            w[it["item_id"]] = (pi / 5) * m[g(it["kind"])] / ng[g(it["kind"])]
    return w


def merge_groups(items, split="test", min_items=8):
    """Kind -> family ("N" or "V") for every family that has a kind with fewer than min_items items in `split`."""
    n = Counter(it["kind"] for it in items if it["split"] == split and it["kind"] in PLANT_KINDS)
    fams = {k[0] for k in PLANT_KINDS if n[k] < min_items}
    return {k: k[0] for k in PLANT_KINDS if k[0] in fams}


def normalize(w):
    w = np.asarray(w, float)
    return w / w.sum()


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def weight_report(df, split="test", min_items=8):
    s = df[(df.split == split) & df.valid & (df.w > 0)]
    nk = s[s.kind.isin(PLANT_KINDS)].kind.value_counts().reindex(PLANT_KINDS, fill_value=0)
    return {"split": split, "n_items": int(len(s)), "n_controls": int((s.kind == "control").sum()), "n_per_kind": {k: int(v) for k, v in nk.items()},
            "min_items_per_kind": int(nk.min()), "kinds_below_min": [k for k in PLANT_KINDS if nk[k] < min_items], "ess": ess(s.w.values) if len(s) else NAN}


# ---------------------------------------------------------------- table, parquet, review pile

STATIC = ("item_id", "entry_key", "header_id", "date", "stratum", "split", "orig_split", "kind", "seed_idx", "truth", "n_lines")
COV = ("second_prose_copy", "n_table_lines", "n_numeric_tokens", "has_threshold_language", "n_chars")


def build(run, splits, pi=PI, tol=TOL):
    items = load_items(run)
    want = [it for it in items if it["split"] in splits]
    calls = load_calls(run, {it["orig_split"] for it in want}, {it["item_id"] for it in want})
    w, wm = weights(items, pi), weights(items, pi, merge_groups(items))
    rows = []
    for it in want:
        r = {k: it.get(k) for k in STATIC}
        r.update({k: it["covariates"].get(k) for k in COV})
        f = features(it, calls.get(it["item_id"], EMPTY()), tol)
        r.update(f)
        r["w"], r["w_merged"] = w[it["item_id"]], wm[it["item_id"]]
        rows.append(r)
    return pd.DataFrame(rows)


def write_review_pile(run, split, items=None):
    """One row per (control entry, line bucket): lines supported by >= 3 of 5 Haiku or >= 2 of 3 Sonnet passes, consecutive lines merged.
    Uses the original split and ignores control_labels (the pile is what gets adjudicated). Blind to model, pass and order."""
    items = items or load_items(run)
    ctrl = [it for it in items if it["kind"] == "control" and it["orig_split"] == split]
    calls = load_calls(run, {split}, {it["item_id"] for it in ctrl})
    rows = []
    for it in ctrl:
        rec = calls.get(it["item_id"])
        if not rec:
            continue
        passes = list(rec["haiku"].values()) + list(rec["sonnet"].values())
        hs, ss = [p["lines"] for p in rec["haiku"].values()], [p["lines"] for p in rec["sonnet"].values()]
        cand = set().union(*(hs + ss)) if hs + ss else set()
        sup = sorted(ln for ln in cand if sum(near(ln, s) for s in hs) >= 3 or sum(near(ln, s) for s in ss) >= 2)
        buckets = []
        for ln in sup:
            if buckets and ln - buckets[-1][-1] <= 1:
                buckets[-1].append(ln)
            else:
                buckets.append([ln])
        table = "\n".join(raw for raw, c in lines_of(it["text"]).values() if c.lstrip().startswith("|"))
        for b in buckets:
            pr = [x for p in passes for x in p["probs"] if b[0] - 1 <= x[0] <= b[-1] + 1]
            quotes = sorted({q for _, q, _ in pr})
            expl = sorted({" ".join(MODEL_WORDS.sub("", e).split()) for _, _, e in pr})
            bl = ",".join(map(str, b))
            rid = hashlib.sha1(f"ar3-review|{it['entry_key']}|{bl}".encode()).hexdigest()[:10]
            rows.append({"row_id": rid, "entry_key": it["entry_key"], "header_id": it["header_id"], "bucket_lines": bl,
                         "quotes": " || ".join(quotes), "explanations": " || ".join(expl), "table_lines": table, "label": ""})
    rows.sort(key=lambda r: r["row_id"])
    path = Path(run) / f"review_pile_{split}.csv"
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["row_id", "entry_key", "header_id", "bucket_lines", "quotes", "explanations", "table_lines", "label"])
        wr.writeheader()
        wr.writerows(rows)
    return path, len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["burn", "test", "probe", "all"])
    ap.add_argument("--run-dir", default=str(RUN))
    ap.add_argument("--pi", type=float, default=PI)
    ap.add_argument("--tol", type=lambda s: s if s == "any" else int(s), default=TOL, help="localization tolerance: 0, 1 (default, stored accept set), 3, or any")
    ap.add_argument("--pile", action="store_true", help="also write review_pile_<split>.csv for each requested burn/test split")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.split:
        ap.error("--split is required")
    run = Path(a.run_dir)
    splits = ["burn", "test", "probe"] if a.split == "all" else [a.split]
    if "test" in splits:
        check_frozen(run)  # refuse up front, whether or not any test item or call exists
    df = build(run, splits, a.pi, a.tol)
    if df.empty:
        raise SystemExit(f"no items in split(s) {splits} in {run / 'items.jsonl'}; nothing written")
    df.to_parquet(run / "items.parquet", index=False)
    print(f"wrote {run / 'items.parquet'}: {len(df)} rows, splits {splits}, tol {a.tol}, pi {a.pi}")
    for s in splits:
        d = df[df.split == s]
        print(f"{s}: {len(d)} items, {int(d.valid.sum())} valid, kinds {d.kind.value_counts().to_dict()}")
        if s != "probe" and d.valid.any():
            print("  weights:", json.dumps(weight_report(df, s)))
    if a.pile:
        for s in (x for x in splits if x != "probe"):
            p, n = write_review_pile(run, s)
            print(f"wrote {p}: {n} rows")


# ---------------------------------------------------------------- selftest


def selftest():
    close = lambda x, y: abs(x - y) < 1e-9
    isnan = lambda x: isinstance(x, float) and math.isnan(x)

    # sim edge cases
    assert sim(set(), set()) == 1.0
    assert sim(set(), {5}) == 0.0 and sim({5}, set()) == 0.0
    assert sim({5, 9}, {5, 9}) == 1.0
    assert sim({5}, {6}) == 1.0 and sim({6}, {5}) == 1.0  # +-1 matches
    assert sim({5}, {7}) == 0.0  # distance 2 does not
    assert close(sim({5, 7}, {6}), 2 / 3)  # one match only (one-to-one)
    assert close(sim({1, 2, 3}, {2}), 0.5)
    assert close(sim({4}, {3, 5}), 2 / 3)
    assert sim({2, 4}, {1, 3}) == 1.0  # lower neighbour first, both matched
    assert close(sim({3, 4}, {3}), 2 / 3)  # exact match consumed first
    assert close(sim({3, 4}, {3, 5}), 1.0)
    assert lines_disagree(set(), set()) is False and lines_disagree({1}, set()) and lines_disagree(set(), {1})
    assert lines_disagree({5}, {6}) is False and lines_disagree({5}, {8}) is True

    # supported lines and accept sets
    assert supported([{10}, {11}, {12}], 2) == {10, 11, 12} and supported([{10}, {12}, set()], 2) == set()
    assert supported([{8}], 1) == {8} and supported([], 1) == set()
    plant = {"lines": [8], "old": "0.50", "new": "0.30", "accept_lines": [5, 6, 7, 8, 9]}
    assert accept_set(plant, 1) == {5, 6, 7, 8, 9}
    assert accept_set(plant, 0) == {5, 6, 8}
    assert accept_set(plant, 3) == set(range(3, 12))
    assert judge("clean", None, set()) == (False, True, False) and judge("clean", None, {4}) == (True, False, False)
    assert judge("N-big", plant, {9}) == (True, True, False)
    assert judge("N-big", plant, {10}) == (True, False, True) and judge("N-big", plant, {10}, 3) == (True, True, False)
    assert judge("N-big", plant, {7}, 0) == (True, False, True) and judge("N-big", plant, {20}, "any") == (True, True, False)
    assert judge("N-big", plant, set()) == (False, False, False)  # miss, not flagged_elsewhere
    assert judge("natural", None, {1}) == (True, None, None)

    # weights, normalization, ESS (6 pool kinds fabricated: 8 controls, N-big 3, N-mid 2, N-stale 2, V-body 1, V-coh 0)
    def mk(i, kind, split="test", truth=None):
        return {"item_id": f"w{i}", "kind": kind, "split": split, "truth": truth or ("clean" if kind == "control" else kind)}
    pool = [mk(i, "control") for i in range(8)] + [mk(10 + i, "N-big") for i in range(3)] + [mk(20 + i, "N-mid") for i in range(2)] \
        + [mk(30 + i, "N-stale") for i in range(2)] + [mk(40, "V-body")]
    pool += [mk(50, "control", "probe", "natural"), mk(51, "control", "burn", "natural"), mk(52, "N-big", "probe")]  # probe rows and natural controls get 0
    pool[-2]["split"] = "probe"
    w = weights(pool)
    assert close(w["w0"], 0.75 / 8) and close(w["w10"], 0.05 / 3) and close(w["w20"], 0.025) and close(w["w40"], 0.05)
    assert w["w50"] == 0 and w["w51"] == 0 and w["w52"] == 0
    assert close(sum(w.values()), 0.95)  # V-coh has no items, its 0.05 is unassigned until renormalized
    vals = [v for v in w.values() if v > 0]
    assert close(normalize(vals).sum(), 1.0) and len(vals) == 16
    exp_ess = 0.95 ** 2 / (8 * (0.75 / 8) ** 2 + 3 * (0.05 / 3) ** 2 + 4 * 0.025 ** 2 + 0.05 ** 2)
    assert close(ess(vals), exp_ess) and abs(exp_ess - 11.852) < 0.01
    assert close(ess([1, 1, 1, 1]), 4.0) and close(ess([1, 1, 2]), 16 / 6) and close(ess([3, 3]), 2.0)
    mg = merge_groups(pool, "test")
    assert mg == {k: k[0] for k in PLANT_KINDS}  # every kind has fewer than 8 test items: both families merge
    wm = weights(pool, groups=mg)
    assert close(wm["w10"], 0.05 * 3 / 7) and close(wm["w40"], 0.05 * 2 / 1) and close(sum(wm.values()), 1.0)
    assert merge_groups([mk(i, "N-big") for i in range(9)] + [mk(20 + i, "N-mid") for i in range(8)] + [mk(40 + i, "N-stale") for i in range(8)]
                        + [mk(60 + i, "V-body") for i in range(8)] + [mk(80, "V-coh")], "test") == {"V-body": "V", "V-coh": "V"}

    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        text = "\n".join(f"L{i + 1:03d}| {t}" for i, t in enumerate([
            "## E header", "", "| cfg | acc |", "|---|---|", "| a | 0.50 |", "| b | 0.50 |", "Prose intro.",
            "Accuracy was 0.30 on cfg a.", "Another prose line.", "Verdict: pass (threshold 0.4)", "", "Tail."]))
        cov = {"second_prose_copy": None, "n_table_lines": 3, "n_numeric_tokens": 4, "has_threshold_language": True, "n_chars": 400}

        def item(iid, ek, kind, split, plant=None):
            return {"item_id": iid, "entry_key": ek, "header_id": "H-" + ek, "date": "2026-10-01", "stratum": "oracle", "split": split, "kind": kind,
                    "seed_idx": 0, "text": text, "n_lines": 12, "plant": plant, "covariates": cov}

        def call(iid, alias, idx, lines, conf=80, ok=True, quote="Accuracy was 0.30", split="burn"):
            so = {"problems": [{"line": ln, "kind": "number", "quote": quote, "explanation": "Haiku says mismatch"} for ln in lines], "judgment_confidence": conf}
            return {"item_id": iid, "split": split, "model_alias": alias, "pass_idx": idx, "ok": ok, "structured_output": so if ok else None,
                    "total_cost_usd": 0.01 if alias == "haiku" else 0.05}

        items = [item("A", "E1", "N-big", "burn", plant), item("B", "E1", "control", "burn"), item("C", "E2", "control", "burn"),
                 item("D", "E2", "N-mid", "burn", plant), item("F", "E3", "control", "burn"), item("G", "E4", "N-big", "test", plant),
                 item("H", "E4", "control", "test"), item("P", "E5", "control", "probe")]
        (run / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
        (run / "control_labels.json").write_text(json.dumps({"E2": "natural"}))
        quote_far = "Accuracy was 0.30"
        cb = [  # item A: planted N-big at line 8, accept 5..9
            call("A", "haiku", 1, [8], 80), call("A", "haiku", 2, [9], 60), call("A", "haiku", 3, [12], 40), call("A", "haiku", 4, [], 90),
            call("A", "haiku", 5, [8], 70), call("A", "sonnet", 1, [8]), call("A", "sonnet", 2, [20]), call("A", "sonnet", 3, [8]),
            # item B: clean control, nothing flagged except pass 5
            *[call("B", "haiku", i, [], 95) for i in (1, 2, 3, 4)], call("B", "haiku", 5, [3], 95),
            call("B", "sonnet", 1, []), call("B", "sonnet", 2, []), call("B", "sonnet", 3, [5]),
            # item D: all Haiku flag line 20 (elsewhere), Sonnet finds nothing; the first record of H1 is a failed call, a later duplicate is ignored
            call("D", "haiku", 1, [], ok=False), *[call("D", "haiku", i, [20], 50, quote="nothing here") for i in (1, 2, 3)],
            call("D", "haiku", 1, [], 10),
            *[call("D", "sonnet", i, []) for i in (1, 2, 3)],
            # item F: Haiku pass 3 failed -> invalid
            call("F", "haiku", 1, []), call("F", "haiku", 2, []), call("F", "haiku", 3, [], ok=False), call("F", "sonnet", 1, []), call("F", "sonnet", 2, []),
            # item C (natural control): moved to probe
            *[call("C", "haiku", i, []) for i in (1, 2, 3)], call("C", "sonnet", 1, []), call("C", "sonnet", 2, []),
        ]
        (run / "calls_burn.jsonl").write_text("\n".join(json.dumps(r) for r in cb) + "\n{truncated")
        (run / "calls_test.jsonl").write_text("garbage that burn must never read\n")
        (run / "calls_probe.jsonl").write_text("\n".join(json.dumps(call("P", "haiku", i, [4, 5], split="probe")) for i in (1, 2, 3)) + "\n")

        # burn run: opens only calls_burn.jsonl, does not need frozen.json
        main(["--split", "burn", "--run-dir", td, "--pile"])
        df = pd.read_parquet(run / "items.parquet").set_index("item_id")
        assert set(df.index) == {"A", "B", "D", "F"} and (df.split == "burn").all()
        A, B, D, F = (df.loc[k] for k in "ABDF")
        # item A
        assert A.truth == "N-big" and A.valid and A.n_haiku_valid == 5 and A.n_sonnet_valid == 3
        assert close(A.conf_agree, 1 / 3) and close(A.conf_agree_10, 0.3)
        assert close(A.conf_vote, 1.0) and close(A.conf_verb, 0.6) and close(A.conf_verb1, 0.8)
        assert close(A.mean_n_problems, 1.0) and close(A.flag_frac, 1.0) and close(A.q_ok, 2 / 3)
        assert close(A.log_chars, math.log(400)) and A.n_numeric_tokens == 4 and A.has_threshold_language == 1
        assert A.h1_flag == 1 and A.h1_correct == 1 and list(A.h1_supported) == [8]
        assert A.h3_flag == 1 and A.h3_correct == 1 and A.h3_flagged_elsewhere == 0 and list(A.h3_supported) == [8, 9]
        assert A.h5_flag == 1 and A.h5_correct == 1 and list(A.h5_supported) == [8, 9]
        assert A.s1_correct == 1 and A.s2_correct == 0 and A.s3_correct == 1 and close(A.sonnet_single_expect, 2 / 3)
        assert A.sonnet_majority_flag == 1 and A.sonnet_majority_correct == 1 and list(A.sonnet_majority_lines) == [8]
        assert close(A.sonnet_agree, (0 + 1 + 0) / 3)
        assert A.proxy_error_sonnet == 0 and A.proxy_error_truth == 0
        assert close(A.haiku_cost_mean_usd, 0.01) and close(A.sonnet_cost_mean_usd, 0.05)
        # item B (clean control)
        assert B.truth == "clean" and close(B.conf_agree, 1.0) and close(B.conf_vote, 1.0) and B.flag_frac == 0 and close(B.q_ok, 1.0)
        assert close(B.conf_agree_10, 0.6)  # 6 empty-empty pairs score 1, 4 pairs with pass 5 score 0
        assert B.h1_correct == 1 and B.h3_correct == 1 and B.h5_correct == 1 and list(B.h5_supported) == []
        assert B.sonnet_majority_flag == 0 and B.sonnet_majority_correct == 1 and B.proxy_error_sonnet == 0 and B.proxy_error_truth == 0
        assert B.s3_correct == 0 and close(B.sonnet_single_expect, 2 / 3)
        # item D: confident wrong, flagged elsewhere; failed first record and later duplicate ignored (first valid wins: H1 conf 50)
        assert close(D.conf_agree, 1.0) and close(D.conf_verb, 0.5) and close(D.q_ok, 0.0)
        assert D.h3_flag == 1 and D.h3_correct == 0 and D.h3_flagged_elsewhere == 1 and D.proxy_error_truth == 1
        assert D.sonnet_majority_flag == 0 and D.proxy_error_sonnet == 1 and D.s1_correct == 0 and D.sonnet_single_expect == 0
        assert D.h1_correct == 0 and D.h1_flagged_elsewhere == 1 and isnan(D.conf_agree_10) and isnan(D.h5_correct)
        # item F: invalid (pass 3 failed)
        assert F.valid == False and F.n_haiku_valid == 2 and isnan(F.conf_agree) and isnan(F.proxy_error_sonnet) and isnan(F.proxy_error_truth)  # noqa: E712
        assert F.h1_correct == 1 and isnan(F.h3_correct)
        # weights: pool = burn+test, controls B, F, H clean (C natural excluded), plants A, D, G
        w_ctrl = 0.75 / 3
        assert close(B.w, w_ctrl) and close(F.w, w_ctrl) and close(A.w, 0.05 / 2) and close(D.w, 0.05)  # N-big has A and G, N-mid only D
        assert close(A.w_merged, 0.05) and close(D.w_merged, 0.05)  # all kinds < 8 items: N family merged, 3 kinds * 0.05 / 3 items
        rep = weight_report(df.reset_index(), "burn", min_items=1)
        assert rep["n_items"] == 3 and rep["n_controls"] == 1 and rep["n_per_kind"]["N-big"] == 1 and rep["min_items_per_kind"] == 0
        assert rep["kinds_below_min"] == ["N-stale", "V-body", "V-coh"]
        assert close(rep["ess"], ess([w_ctrl, 0.025, 0.05]))

        # review pile (burn): control rows only, ignores control_labels, blind, deterministic ids
        pile = pd.read_csv(run / "review_pile_burn.csv", dtype=str, keep_default_na=False)
        assert list(pile.columns) == ["row_id", "entry_key", "header_id", "bucket_lines", "quotes", "explanations", "table_lines", "label"]
        assert (pile.label == "").all()
        assert not pile.explanations.str.contains("haiku|sonnet", case=False).any()
        # control B: only 1 Haiku pass (of 5) flags line 3 and 1 Sonnet (of 3) flags line 5 -> no bucket; C and F none -> pile empty
        assert len(pile) == 0

        # test split is closed without a valid freeze
        for bad in ("none", "mismatch"):
            if bad == "mismatch":
                (run / "frozen.json").write_text("{}")
                (run / "frozen.sha256").write_text("0" * 64 + "  frozen.json\n")
            try:
                main(["--split", "test", "--run-dir", td])
                raise AssertionError("test split opened without a valid freeze")
            except SystemExit as e:
                assert "refusing" in str(e)
        try:
            main(["--split", "all", "--run-dir", td])
            raise AssertionError("all opened without a valid freeze")
        except SystemExit:
            pass
        (run / "frozen.sha256").write_text("")  # empty hash file also refuses (no IndexError)
        try:
            main(["--split", "test", "--run-dir", td])
            raise AssertionError("test split opened with an empty frozen.sha256")
        except SystemExit as e:
            assert "refusing" in str(e)
        with tempfile.TemporaryDirectory() as td2:  # no test items at all: the guard still fires first, then an empty split is a clean exit
            Path(td2, "items.jsonl").write_text("")
            for sp, msg in (("test", "refusing"), ("burn", "no items")):
                try:
                    main(["--split", sp, "--run-dir", td2])
                    raise AssertionError("empty split did not exit")
                except SystemExit as e:
                    assert msg in str(e)
        (run / "frozen.sha256").write_text(hashlib.sha256((run / "frozen.json").read_bytes()).hexdigest() + "  frozen.json\n")
        (run / "calls_test.jsonl").write_text("\n".join(json.dumps(r) for r in [
            *[call("G", "haiku", i, [8], split="test") for i in (1, 2, 3)], call("G", "sonnet", 1, [8], split="test"), call("G", "sonnet", 2, [8], split="test"),
            *[call("H", "haiku", i, [], split="test") for i in (1, 2, 3)], call("H", "sonnet", 1, [], split="test"), call("H", "sonnet", 2, [], split="test")]) + "\n")
        main(["--split", "test", "--run-dir", td])
        dt = pd.read_parquet(run / "items.parquet").set_index("item_id")
        assert set(dt.index) == {"G", "H"} and dt.loc["G"].h3_correct == 1 and dt.loc["H"].h3_correct == 1 and dt.loc["G"].valid
        assert dt.loc["G"].n_haiku_valid == 3 and isnan(dt.loc["G"].conf_agree_10) and dt.loc["G"].h5_correct != dt.loc["G"].h5_correct
        assert close(dt.loc["G"].sonnet_agree, 1.0) and dt.loc["G"].proxy_error_sonnet == 0

        # probe: natural control C (moved from burn, truth natural, correctness unknown) and the original probe control P (truth unknown)
        main(["--split", "probe", "--run-dir", td])  # needs calls_burn (frozen is valid anyway) and calls_probe
        dp = pd.read_parquet(run / "items.parquet").set_index("item_id")
        assert set(dp.index) == {"C", "P"} and dp.loc["C"].truth == "natural" and dp.loc["P"].truth == "unknown"
        assert isnan(dp.loc["C"].h3_correct) and isnan(dp.loc["P"].h3_correct) and dp.loc["C"].h3_flag == 0 and dp.loc["P"].h3_flag == 1
        assert dp.loc["C"].w == 0 and dp.loc["P"].w == 0 and isnan(dp.loc["C"].proxy_error_truth)

        # review pile with a real bucket (test split, after freeze): 3 of 5 Haiku passes near line 6, 2 of 3 Sonnet passes at line 11
        (run / "calls_test.jsonl").write_text("\n".join(json.dumps(r) for r in [
            call("H", "haiku", 1, [6], quote="b 0.50", split="test"), call("H", "haiku", 2, [7], quote="Prose intro.", split="test"),
            call("H", "haiku", 3, [], split="test"), call("H", "haiku", 4, [6], quote="b 0.50", split="test"), call("H", "haiku", 5, [], split="test"),
            call("H", "sonnet", 1, [11], quote="Tail", split="test"), call("H", "sonnet", 2, [12], quote="Tail", split="test"), call("H", "sonnet", 3, [], split="test"),
            call("G", "haiku", 1, [6], split="test")]) + "\n")
        path, n = write_review_pile(run, "test")
        pile = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert n == 2 and list(pile.entry_key) == ["E4", "E4"] and set(pile.bucket_lines) == {"6,7", "11,12"}  # G is a plant, never in the pile
        b1 = pile[pile.bucket_lines == "6,7"].iloc[0]
        assert b1.quotes == "Prose intro. || b 0.50" and b1.explanations == "says mismatch" and b1.header_id == "H-E4" and b1.label == ""
        assert "L005| | a | 0.50 |" in b1.table_lines and "L004| |---|---|" in b1.table_lines and "L008|" not in b1.table_lines
        assert pile.row_id.is_unique and (pile.row_id == pile.row_id.sort_values()).all()

        # control_labels: bad value rejected; natural moves a control to probe, plants stay
        (run / "control_labels.json").write_text(json.dumps({"E1": "maybe"}))
        try:
            load_items(run)
            raise AssertionError("bad label accepted")
        except ValueError:
            pass
        (run / "control_labels.json").unlink()
        it2 = {i["item_id"]: i for i in load_items(run)}
        assert it2["C"]["split"] == "burn" and it2["C"]["truth"] == "clean" and it2["D"]["split"] == "burn"  # default: all controls clean

        # tolerance sensitivities flow through build (needs the burn file only)
        for tol, exp in ((0, 1.0), (1, 1.0), (3, 1.0), ("any", 1.0)):
            assert build(run, ["burn"], tol=tol).set_index("item_id").loc["A"].h3_correct == exp
        assert build(run, ["burn"], tol="any").set_index("item_id").loc["D"].h3_correct == 1  # any flag counts
        assert build(run, ["burn"], tol=3).set_index("item_id").loc["D"].h3_correct == 0  # line 20 is far from 3..11
    print("selftest passed")


if __name__ == "__main__":
    main()
