"""Run the avrl_router diagnostic on every saved point and build a comparison table.

Not part of the avrl_router package; a one-off validation driver for v0.
Run from /homes/rdubash/active-audit/avrl with the project venv.
"""

import json
import os
import subprocess
import sys

AVRL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../avrl
PYTHON = "/homes/rdubash/active-audit/.venv/bin/python"
OUT_ROOT = os.path.join(AVRL_DIR, "runs", "diag_v0")
BUDGETS = "0.1,0.25,0.5"
TARGET_CORRUPTION = "0.02"

# (point id, groups file relative to avrl/, confidence columns, note if unavailable)
POINTS = [
    ("E03", None, None, "no feats.parquet/groups.parquet; only result.json (a precomputed experiment summary) exists, no row-level confidence+error table"),
    ("E04", "runs/E04/judge_scores.parquet", None, "has judge_scores.parquet but no agreement/rm_gap12/consistency column; judge_pick_share and judge_error measure something else (judge-vs-gold, not cheap-vs-expensive disagreement) and would need to be invented into this shape, which the spec forbids"),
    ("E05", "runs/E05/selfverify.parquet", None, "has selfverify.parquet with sv_agreement/self_verify_error, a self-verify audit, not the agreement/proxy_error groups schema; skipped rather than reinterpreted"),
    ("E06", "runs/E06/feats.parquet", ["agreement"], None),
    ("E08", "runs/E08/feats.parquet", ["agreement"], None),
    ("E09", "runs/E09/feats.parquet", ["rm_gap12", "consistency"], None),
    ("E09b", "runs/E09b/feats.parquet", ["rm_gap12", "consistency"], None),
    ("E09c", "runs/E09c/feats.parquet", ["rm_gap12", "consistency"], None),
    ("E09d", "runs/E09d/feats.parquet", ["rm_gap12", "consistency"], None),
    ("p0", "runs/p0/feats.parquet", ["agreement"], None),
]


def run_point(point_id, groups_rel, confidence_cols):
    out_dir = os.path.join(OUT_ROOT, point_id)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        PYTHON, "-m", "avrl_router.diagnose",
        "--groups", groups_rel,
        "--confidence", ",".join(confidence_cols),
        "--error", "proxy_error",
        "--budgets", BUDGETS,
        "--target_corruption", TARGET_CORRUPTION,
        "--out", out_dir,
    ]
    proc = subprocess.run(cmd, cwd=AVRL_DIR, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"{point_id} failed:\n{proc.stdout}\n{proc.stderr}")
    with open(os.path.join(out_dir, "diagnosis.json")) as f:
        return json.load(f)


def main():
    rows = []
    results = {}
    for point_id, groups_rel, confidence_cols, note in POINTS:
        if confidence_cols is None:
            rows.append({"point": point_id, "confidence_col": None, "note": note})
            results[point_id] = {"available": False, "note": note}
            continue
        result = run_point(point_id, groups_rel, confidence_cols)
        results[point_id] = {"available": True, "result": result}
        n = result["n_groups"]
        err = result["proxy_error_rate"]["point"]
        for col in confidence_cols:
            c = result["confidence_columns"][col]
            b25 = c["budgets"]["0.25"]
            rows.append({
                "point": point_id,
                "confidence_col": col,
                "n": n,
                "proxy_error_rate": round(err, 4),
                "auc": None if c["auc"]["point"] != c["auc"]["point"] else round(c["auc"]["point"], 4),
                "corrupted_threshold_25pct": round(b25["threshold"]["point"], 4),
                "corrupted_random_25pct": round(b25["random"]["point"], 4),
                "relative_reduction_25pct": round(b25["relative_reduction_vs_random"], 4),
                "recommended_budget_target_0.02": c["recommended_budget"],
                "regime": c["regime"],
                "note": None,
            })

    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump({"rows": rows, "points": results}, f, indent=2)

    header = ["point", "confidence_col", "n", "proxy_error_rate", "auc", "corrupted_thresh@25%", "corrupted_random@25%", "rel_reduction@25%", "recommended_budget(target=0.02)", "regime"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        if r["confidence_col"] is None:
            lines.append(f"| {r['point']} | unavailable | - | - | - | - | - | - | - | {r['note']} |")
        else:
            lines.append(
                f"| {r['point']} | {r['confidence_col']} | {r['n']} | {r['proxy_error_rate']} | {r['auc']} | "
                f"{r['corrupted_threshold_25pct']} | {r['corrupted_random_25pct']} | {r['relative_reduction_25pct']} | "
                f"{r['recommended_budget_target_0.02']} | {r['regime']} |"
            )
    table = "\n".join(lines)

    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("# avrl_router diagnostic v0: validation across saved points\n\n")
        f.write(f"Budgets reported: {BUDGETS}. Target corruption for recommended budget: {TARGET_CORRUPTION}. n_boot=1000, seed=0.\n\n")
        f.write(table + "\n")

    print(table)


if __name__ == "__main__":
    main()
