"""
Figure 4 for the AVRL thesis: the common re-evaluation of final adapters.
Five panels in one row, one bar-plus-seed-markers group per arm: GSM8K
accuracy, an open-ended 3B-judge panel, an open-ended 7B-judge (hannah)
panel, an open-ended 4-bit 7B-judge (goldberg) panel, and MBPP pass@1 on
the full 500-problem test set.

Every number plotted here is read directly from the re-evaluation tables
under avrl/runs/reeval/ (one JSON line per adapter run, written by the
common re-eval harness), not copied from RESEARCH_LOG.md prose. Missing
rows (a table still growing) are skipped with a printed note, not filled
in. Re-run this script if the underlying JSONL files change; do not
hand-edit the PNG/PDF outputs or FIGURE4.md's data table.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REEVAL_DIR = "/homes/rdubash/active-audit/avrl/runs/reeval"
OUT_DIR = "/homes/rdubash/active-audit/lit/avrl/fig"

# Same palette as make_fig2.py: color marks mechanism, not task.
COLOR_LABELFREE = "#2a78d6"  # blue: label-free rule (threshold, auto), no refitting
COLOR_LEARNED = "#eb6834"    # orange: learned allocator (greedy, bandit), refits on bought labels
COLOR_REFERENCE = "#8a8a8a"  # grey: full / proxy / random reference arms, plus untrained base
GRID_COLOR = "#dddddd"
TEXT_COLOR = "#222222"

# (arm_key, plain-word label, kind, {seed: run_name})
PANEL_A_ARMS = [
    ("thr25", "threshold 25%", "labelfree",
     {0: "m07_threshold_s0", 1: "m07_threshold_s1"}),
    ("thr10", "threshold 10%", "labelfree",
     {0: "m14_threshold_b10_s0", 1: "m14_threshold_b10_s1"}),
    ("greedy", "greedy (self-train)", "learned",
     {0: "m07_greedy_s0", 1: "m07_greedy_s1"}),
    ("learned", "learned (bandit)", "learned",
     {0: "m07_learned_s0", 1: "m07_learned_s1"}),
    ("full", "full (800 calls)", "reference",
     {0: "m10_full_s0", 1: "m10_full_s1"}),
    ("proxy", "proxy (0 calls)", "reference",
     {0: "m10_proxy_s0", 1: "m10_proxy_s1"}),
]

PANEL_B_ARMS = [
    ("thrcons", "threshold (consistency)", "labelfree",
     {0: "p1_judge_thrcons_lr5_s0", 1: "p1_judge_thrcons_lr5_s1", 3: "p1_judge_thrcons_lr5_s3"}),
    ("threshold_rm", "threshold (RM gap)", "labelfree",
     {0: "p1_judge_threshold_lr5", 1: "p1_judge_threshold_lr5_s1"}),
    ("greedy", "greedy (self-train)", "learned",
     {0: "p1_judge_greedy_lr5_s0", 1: "p1_judge_greedy_lr5_s1"}),
    ("learned", "learned (bandit)", "learned",
     {0: "p1_judge_learned_lr5", 1: "p1_judge_learned_lr5_s1"}),
    ("random", "random", "reference",
     {0: "p1_judge_random_lr5_s0", 1: "p1_judge_random_lr5_s1"}),
    ("full", "full (800 calls)", "reference",
     {0: "p1_judge_full_lr5_s0", 1: "p1_judge_full_lr5_s1", 3: "p1_judge_full_lr5_s3"}),
    ("full_rank", "full, rank reward", "reference",
     {0: "p1_judge_fullrank_lr5_s0", 1: "p1_judge_fullrank_lr5_s1"}),
]

PANEL_E_ARMS = [
    ("threshold", "threshold", "labelfree",
     {0: "e10b_threshold_s0", 1: "e10b_threshold_s1", 2: "e10b_threshold_s2"}),
    ("random", "random", "reference",
     {0: "e10b_random_s0", 1: "e10b_random_s1", 2: "e10b_random_s2"}),
    ("full", "full", "reference",
     {0: "e10b_full_s0", 1: "e10b_full_s1"}),
    ("proxy", "proxy", "reference",
     {0: "e10b_proxy_s0", 1: "e10b_proxy_s1"}),
    ("proxy600", "proxy (600 steps)", "reference",
     {1: "e10c_proxy600_s1"}),
    ("auto", "auto (abs 2%)", "labelfree",
     {1: "e10e_auto_s1"}),
    ("auto_up", "auto (rel 25% upper)", "labelfree",
     {1: "e10e_up_auto_s1"}),
    ("auto_rel", "auto (rel 25%)", "labelfree",
     {1: "e10e_rel_auto_s1"}),
]

# Panel C (judge320_hannah.jsonl + judge320_hannah_s2.jsonl) has no
# reliable seed field, so order matters instead; see load_hannah_panel.
PANEL_C_ORDER = ["thrcons", "greedy", "random", "full"]
PANEL_C_LABELS = dict(
    thrcons=("threshold (consistency)", "labelfree"),
    greedy=("greedy (self-train)", "learned"),
    random=("random", "reference"),
    full=("full", "reference"),
)

# Panel D: open-ended, 4-bit 7B judge (goldberg). judge320_5090b.jsonl
# holds seeds 0-1 (plus the single-seed threshold_rm/learned arms);
# judge320_m15.jsonl adds seeds 2-3 for thrcons and greedy, by run name.
PANEL_D_ARMS = [
    ("thrcons", "threshold (consistency)", "labelfree",
     {0: "p1_judge5090b_thrcons_s0", 1: "p1_judge5090b_thrcons_s1",
      2: "p1_judge5090b_thrcons_s2", 3: "p1_judge5090b_thrcons_s3"}),
    ("threshold_rm", "threshold (RM gap)", "labelfree",
     {0: "p1_judge5090b_threshold"}),
    ("greedy", "greedy (self-train)", "learned",
     {0: "p1_judge5090b_greedy_s0", 1: "p1_judge5090b_greedy_s1",
      2: "p1_judge5090b_greedy_s2", 3: "p1_judge5090b_greedy_s3"}),
    ("learned", "learned (bandit)", "learned",
     {0: "p1_judge5090b_learned"}),
]


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print(f"  SKIP missing file: {path}")
        return rows
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  SKIP bad json line {i} in {path}: {e}")
    return rows


def get_metric(rows, run_name, metric_key, context):
    for r in rows:
        if r.get("run") == run_name:
            if metric_key not in r:
                print(f"  SKIP {context}{run_name}: missing field {metric_key}")
                return None
            return r[metric_key]
    print(f"  SKIP {context}{run_name}: not found yet (table may still be growing)")
    return None


def load_simple_panel(filename, metric_key, arms_spec):
    """Panels A, B, and D: one row per run, looked up by run name.
    filename may be a single jsonl file or a list of files to merge
    (the base row is read from the first file only)."""
    filenames = [filename] if isinstance(filename, str) else list(filename)
    rows = load_jsonl(os.path.join(REEVAL_DIR, filenames[0]))
    for extra in filenames[1:]:
        rows.extend(r for r in load_jsonl(os.path.join(REEVAL_DIR, extra)) if r.get("run") != "base")
    base_row = next((r for r in rows if r.get("run") == "base"), None)
    base_val = base_row.get(metric_key) if base_row else None
    if base_val is None:
        print(f"  SKIP base in {filenames[0]}: not found")

    arms_out = []
    for key, label, kind, seed_runs in arms_spec:
        seeds = {}
        for seed, run_name in sorted(seed_runs.items()):
            val = get_metric(rows, run_name, metric_key, context=f"{filenames[0]}/")
            if val is not None:
                seeds[seed] = val
        if seeds:
            arms_out.append(dict(key=key, label=label, kind=kind, seeds=seeds))
        else:
            print(f"  SKIP arm '{key}' in {filenames[0]}: no seeds available")
    return base_val, arms_out


def load_code_panel(filename, arms_spec):
    """Panel E: like load_simple_panel, but also averages humaneval_pass1
    per arm for the small text printed above each bar."""
    rows = load_jsonl(os.path.join(REEVAL_DIR, filename))
    # hannah's seed 0 adapters live in a second file with unsuffixed run names; tag them _s0
    for extra in ("code_hannah_e10b.jsonl", "code_e10e_up.jsonl"):
        extra_path = os.path.join(REEVAL_DIR, extra)
        if os.path.exists(extra_path):
            for r in load_jsonl(extra_path):
                if r.get("run") != "base":
                    r = dict(r)
                    if extra == "code_hannah_e10b.jsonl":
                        r["run"] = r["run"] + "_s0"
                    rows.append(r)
    base_row = next((r for r in rows if r.get("run") == "base"), None)
    base_mbpp = base_row.get("mbpp_pass1") if base_row else None
    if base_mbpp is None:
        print(f"  SKIP base in {filename}: not found")

    arms_out = []
    for key, label, kind, seed_runs in arms_spec:
        seeds = {}
        he_vals = []
        for seed, run_name in sorted(seed_runs.items()):
            mbpp = get_metric(rows, run_name, "mbpp_pass1", context=f"{filename}/")
            he = get_metric(rows, run_name, "humaneval_pass1", context=f"{filename}/")
            if mbpp is not None:
                seeds[seed] = mbpp
            if he is not None:
                he_vals.append(he)
        if seeds:
            mean_he = sum(he_vals) / len(he_vals) if he_vals else None
            arms_out.append(dict(key=key, label=label, kind=kind, seeds=seeds, humaneval_mean=mean_he))
        else:
            print(f"  SKIP arm '{key}' in {filename}: no seeds available")
    return base_mbpp, arms_out


def load_hannah_panel():
    """Panel C: judge320_hannah.jsonl has no reliable seed field, so seeds
    are assigned by file and position: the first four non-base rows (in
    file order full, random, thrcons, greedy) are seed 0, the next four
    are seed 1. judge320_hannah_s2.jsonl holds one more row (full) as
    seed 2."""
    f1 = os.path.join(REEVAL_DIR, "judge320_hannah.jsonl")
    f2 = os.path.join(REEVAL_DIR, "judge320_hannah_s2.jsonl")
    rows1 = load_jsonl(f1)
    rows2 = load_jsonl(f2)
    # later seeds: one file per seed, same run names, seed given by the file suffix
    extra_seed_files = [(3, os.path.join(REEVAL_DIR, "judge320_hannah_s3.jsonl")),
                        (4, os.path.join(REEVAL_DIR, "judge320_hannah_s4.jsonl"))]

    base_row = next((r for r in rows1 if r.get("run") == "base"), None)
    base_val = base_row.get("judge_score_mean") if base_row else None
    if base_val is None:
        print(f"  SKIP base in judge320_hannah.jsonl: not found")

    file_order = ["full", "random", "thrcons", "greedy"]
    non_base1 = [r for r in rows1 if r.get("run") != "base"]
    seeds = {name: {} for name in PANEL_C_ORDER}
    for i, r in enumerate(non_base1):
        name = file_order[i % 4]
        seed = i // 4
        if "judge_score_mean" in r:
            seeds[name][seed] = r["judge_score_mean"]
        else:
            print(f"  SKIP judge320_hannah.jsonl row {i}: missing judge_score_mean")
    if len(non_base1) % 4 != 0:
        print(f"  NOTE judge320_hannah.jsonl: {len(non_base1)} non-base rows "
              f"(not a multiple of 4), last seed block may be partial")

    non_base2 = [r for r in rows2 if r.get("run") != "base"]
    for r in non_base2:
        name = r.get("run")
        if name in seeds and "judge_score_mean" in r:
            seeds[name][2] = r["judge_score_mean"]
        else:
            print(f"  SKIP judge320_hannah_s2.jsonl row run={name}: unexpected run or missing metric")

    for extra_seed, fpath in extra_seed_files:
        if not os.path.exists(fpath):
            continue
        for r in load_jsonl(fpath):
            name = r.get("run")
            if name == "base" or name not in seeds:
                continue
            if "judge_score_mean" in r:
                seeds[name][extra_seed] = r["judge_score_mean"]

    arms_out = []
    for key in PANEL_C_ORDER:
        if seeds[key]:
            label, kind = PANEL_C_LABELS[key]
            arms_out.append(dict(key=key, label=label, kind=kind, seeds=seeds[key]))
        else:
            print(f"  SKIP arm '{key}' in judge320_hannah: no seeds available")
    return base_val, arms_out


def style_axes(ax):
    ax.grid(True, axis="y", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)


def seed_offsets(n_seeds):
    if n_seeds <= 1:
        return [0.0]
    if n_seeds == 2:
        return [-0.09, 0.09]
    return [-0.12, 0.0, 0.12]


def set_ylim_tight(ax, values, bottom_pad=0.15, top_pad=0.15):
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng <= 0:
        rng = max(abs(hi), 0.01) * 0.1
    ax.set_ylim(lo - bottom_pad * rng, hi + top_pad * rng)


def draw_panel(ax, base_val, arms, y_label, annotate=None, top_pad=0.18):
    x_positions = list(range(len(arms)))
    all_values = []
    bar_tops = []
    for x, arm in zip(x_positions, arms):
        color = {"labelfree": COLOR_LABELFREE, "learned": COLOR_LEARNED}.get(arm["kind"], COLOR_REFERENCE)
        seed_items = sorted(arm["seeds"].items())
        values = [v for _, v in seed_items]
        all_values.extend(values)
        mean_val = sum(values) / len(values)
        bar_tops.append(mean_val)

        ax.bar(
            [x], [mean_val], width=0.5, color=color, alpha=0.32,
            edgecolor=color, linewidth=1.2, zorder=1,
        )
        offsets = seed_offsets(len(seed_items))
        for (seed, v), dx in zip(seed_items, offsets):
            ax.scatter(
                [x + dx], [v], s=52, facecolors=color,
                edgecolors="#333333", linewidths=0.7, alpha=0.95, zorder=3,
            )

    if base_val is not None:
        all_values.append(base_val)
        ax.axhline(base_val, color=COLOR_REFERENCE, linestyle="--", linewidth=1.3, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [a["label"] for a in arms], fontsize=8,
        rotation=30, ha="right", rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel(y_label, fontsize=9)
    style_axes(ax)
    set_ylim_tight(ax, all_values, top_pad=top_pad)

    if annotate is not None:
        y0, y1 = ax.get_ylim()
        for x, top, text in zip(x_positions, bar_tops, annotate):
            if text:
                ax.text(x, top + 0.035 * (y1 - y0), text, ha="center",
                        fontsize=6.3, color=TEXT_COLOR)


def legend_handles():
    labelfree_h = plt.Line2D(
        [0], [0], marker="s", linestyle="none", markersize=10,
        markerfacecolor=COLOR_LABELFREE, markeredgecolor=COLOR_LABELFREE,
        alpha=0.8, label="Label-free rule (threshold, auto)",
    )
    learned_h = plt.Line2D(
        [0], [0], marker="s", linestyle="none", markersize=10,
        markerfacecolor=COLOR_LEARNED, markeredgecolor=COLOR_LEARNED,
        alpha=0.8, label="Learned allocator (greedy / bandit)",
    )
    reference_h = plt.Line2D(
        [0], [0], marker="s", linestyle="none", markersize=10,
        markerfacecolor=COLOR_REFERENCE, markeredgecolor=COLOR_REFERENCE,
        alpha=0.8, label="Reference (full, proxy, random)",
    )
    base_h = plt.Line2D(
        [0], [0], color=COLOR_REFERENCE, linestyle="--", linewidth=1.3,
        label="Untrained base model",
    )
    return [labelfree_h, learned_h, reference_h, base_h]


FOOTNOTE = (
    "Common re-evaluation of final adapters: fixed held-out subsets (eval seed 0), "
    "320-token responses and one 4-bit 3B judge for the open-ended panels, full MBPP "
    "test set for code. Each marker is one seed; the bar is the seed mean."
)


def build_data():
    data = {}
    print("Panel A (GSM8K accuracy):")
    data["A"] = load_simple_panel("gsm8k.jsonl", "accuracy", PANEL_A_ARMS)
    print("Panel B (open-ended, 3B judge):")
    data["B"] = load_simple_panel("judge320.jsonl", "judge_score_mean", PANEL_B_ARMS)
    print("Panel C (open-ended, 7B judge, hannah):")
    data["C"] = load_hannah_panel()
    print("Panel D (open-ended, 4-bit 7B judge, goldberg):")
    data["D"] = load_simple_panel(
        ["judge320_5090b.jsonl", "judge320_m15.jsonl"], "judge_score_mean", PANEL_D_ARMS
    )
    print("Panel E (code, MBPP pass@1, full test set):")
    data["E"] = load_code_panel("code.jsonl", PANEL_E_ARMS)
    return data


def make_figure(data):
    plt.rcParams["font.size"] = 9.5
    fig, axes = plt.subplots(1, 5, figsize=(22.54, 5.4), dpi=200)

    base_a, arms_a = data["A"]
    draw_panel(axes[0], base_a, arms_a, "GSM8K accuracy\n(300 held-out prompts)")
    axes[0].set_title("A. GSM8K accuracy\n(Qwen2.5 1.5B)", fontsize=9.5, color=TEXT_COLOR, pad=8)

    base_b, arms_b = data["B"]
    draw_panel(axes[1], base_b, arms_b, "Judge score\n(320-tok, 4-bit 3B judge)")
    axes[1].set_title("B. Open-ended,\n3B judge panel", fontsize=9.5, color=TEXT_COLOR, pad=8)

    base_c, arms_c = data["C"]
    draw_panel(axes[2], base_c, arms_c, "Judge score\n(320-tok, 4-bit 3B judge)")
    axes[2].set_title("C. Open-ended,\n7B judge panel (hannah)", fontsize=9.5, color=TEXT_COLOR, pad=8)

    base_d, arms_d = data["D"]
    draw_panel(axes[3], base_d, arms_d, "Judge score\n(320-tok, 4-bit 7B judge)")
    axes[3].set_title("D. Open-ended,\n4-bit 7B judge (goldberg)", fontsize=9.5, color=TEXT_COLOR, pad=8)

    base_e, arms_e = data["E"]
    annotate = [
        f"HE {a['humaneval_mean'] * 100:.1f}%" if a.get("humaneval_mean") is not None else None
        for a in arms_e
    ]
    draw_panel(axes[4], base_e, arms_e, "MBPP pass@1\n(500-problem test set)",
               annotate=annotate, top_pad=0.34)
    axes[4].set_title("E. Code, MBPP pass@1\n(full test set)", fontsize=9.5, color=TEXT_COLOR, pad=8)

    fig.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, fontsize=8.2, frameon=False,
    )
    fig.text(0.5, 0.01, FOOTNOTE, ha="center", fontsize=7.6, color=TEXT_COLOR)
    fig.subplots_adjust(top=0.80, bottom=0.34, left=0.044, right=0.988, wspace=0.45)

    png_path = f"{OUT_DIR}/fig4_reeval.png"
    fig.savefig(png_path, dpi=200)
    fig.savefig(f"{OUT_DIR}/fig4_reeval.pdf")
    plt.close(fig)
    print("wrote", png_path, "(+pdf)")


PANEL_NAMES = dict(
    A="A. GSM8K accuracy",
    B="B. Open-ended, 3B judge panel",
    C="C. Open-ended, 7B judge panel (hannah)",
    D="D. Open-ended, 4-bit 7B judge panel (goldberg)",
    E="E. Code, MBPP pass@1 (full test set)",
)


def data_table_rows(data):
    rows = []
    for key in ["A", "B", "C", "D", "E"]:
        base_val, arms = data[key]
        if base_val is not None:
            rows.append((PANEL_NAMES[key], "base (untrained)", "-", f"{base_val:.4f}", f"{base_val:.4f}"))
        for arm in arms:
            seed_items = sorted(arm["seeds"].items())
            seeds_str = ",".join(str(s) for s, _ in seed_items)
            values_str = ",".join(f"{v:.4f}" for _, v in seed_items)
            mean_val = sum(v for _, v in seed_items) / len(seed_items)
            label_oneline = arm["label"].replace("\n", " ")
            rows.append((PANEL_NAMES[key], label_oneline, seeds_str, values_str, f"{mean_val:.4f}"))
    return rows


def write_figure_md(data):
    rows = data_table_rows(data)
    lines = []
    lines.append("# Figure 4: common re-evaluation of final adapters")
    lines.append("")
    lines.append(
        "Files: `fig4_reeval.png`, `fig4_reeval.pdf`. Generated by `make_fig4.py`, "
        "which reads the common re-evaluation tables directly: "
        "`avrl/runs/reeval/gsm8k.jsonl`, `judge320.jsonl`, `judge320_hannah.jsonl`, "
        "`judge320_hannah_s2.jsonl`, `judge320_5090b.jsonl`, `judge320_m15.jsonl`, "
        "`code.jsonl`. Re-run the script if those files "
        "change; do not hand-edit the PNG/PDF outputs or this table."
    )
    lines.append("")
    lines.append("## Data table")
    lines.append("")
    lines.append("| panel | arm | seeds | values | mean |")
    lines.append("|---|---|---|---|---|")
    for panel, arm, seeds_str, values_str, mean_str in rows:
        lines.append(f"| {panel} | {arm} | {seeds_str} | {values_str} | {mean_str} |")
    lines.append("")

    base_e, arms_e = data["E"]
    he_parts = [
        f"{arm['label'].strip()} {arm['humaneval_mean'] * 100:.1f}%"
        for arm in arms_e if arm.get("humaneval_mean") is not None
    ]
    if he_parts:
        lines.append(
            "Panel E mean HumanEval pass@1 by arm (164-problem test, printed above "
            "each bar in the figure): " + "; ".join(he_parts) + "."
        )
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Every trained arm beats the untrained base model on its held-out metric, "
        "in all five panels. No allocation rule, whether label-free threshold, "
        "learned allocator, or reference, separates from the others on held-out "
        "quality; within each panel the arms cluster together, well inside the gap "
        "from the untrained base."
    )
    lines.append("")
    lines.append(f"Footnote: {FOOTNOTE}")
    lines.append("")

    path = f"{OUT_DIR}/FIGURE4.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    return rows


if __name__ == "__main__":
    print("Building fig4_reeval (common re-evaluation of final adapters):")
    data = build_data()
    make_figure(data)

    print("\nFull data table (panel, arm, seeds, values, mean):")
    for panel, arm, seeds_str, values_str, mean_str in data_table_rows(data):
        print(f"{panel}\t{arm}\t{seeds_str}\t{values_str}\t{mean_str}")

    write_figure_md(data)
